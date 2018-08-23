import gym.spaces
import os
import time
import numpy as np
from mpi4py import MPI
from utils import logger
from utils import dataset
from collections import deque
from algos.trpo_mpi import TRPO
from utils.cmd import arg_parser
from utils.tf_primitives import TfUtil
from utils.misc import set_global_seeds
from utils.math import explained_variance
from optimizers.cg import conjugate_gradient


def fit(
        model,
        env,
        timesteps_per_batch,  # what to train on
        max_kl,
        cg_iters,
        gamma,
        lam,  # advantage estimation
        entcoeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters=3,
        max_timesteps=0,
        max_episodes=0,
        max_iters=0,  # time constraint
        callback=None):
    # Setup losses and stuff
    # ----------------------------------------
    # nworkers = MPI.COMM_WORLD.Get_size()
    # rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)

    th_init = model.get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    model.set_from_flat(th_init)
    model.vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = model.traj_segment_generator(
        model.pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    while True:
        if callback:
            callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************" % iters_so_far)

        with model.timed("sampling"):
            seg = seg_gen.__next__()

        model.add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg[
            "tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()
                 ) / atarg.std()  # standardized advantage function estimate

        if hasattr(model.pi, "ret_rms"):
            model.pi.ret_rms.update(tdlamret)
        if hasattr(model.pi, "ob_rms"):
            model.pi.ob_rms.update(ob)  # update running mean/std for policy

        args = seg["ob"], seg["ac"], atarg
        fvpargs = [arr[::5] for arr in args]

        def fisher_vector_product(p):
            return model.allmean(model.compute_fvp(p, *fvpargs)) + cg_damping * p

        model.assign_old_eq_new()  # set old parameter values to new parameter values
        with model.timed("computegrad"):
            *lossbefore, g = model.compute_lossandgrad(*args)
        lossbefore = model.allmean(np.array(lossbefore))
        g = model.allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with model.timed("cg"):
                stepdir = conjugate_gradient(
                    fisher_vector_product,
                    g,
                    cg_iters=cg_iters,
                    verbose=model.rank == 0
                )
            assert np.isfinite(stepdir).all()
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = model.get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                model.set_from_flat(thnew)
                meanlosses = surr, kl, *_ = model.allmean(
                    np.array(model.compute_losses(*args))
                )
                improve = surr - surrbefore
                logger.log(
                    "Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                model.set_from_flat(thbefore)
            if model.nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather(
                    (thnew.sum(), model.vfadam.getflat().sum()))  # list of tuples
                assert all(
                    np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(model.loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with model.timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches(
                    (seg["ob"], seg["tdlamret"]),
                        include_final_partial_batch=False,
                        batch_size=64):
                    g = model.allmean(model.compute_vflossandgrad(mbob, mbret))
                    model.vfadam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before",
                              explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(model.flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if model.rank == 0:
            logger.dump_tabular()


def main():
    parser = arg_parser()
    parser.add_argument(
        '--platform',
        help='environment choice',
        choices=['atari', 'mujoco'],
        default='atari')

    platform_args, environ_args = parser.parse_known_args()
    platform = platform_args.platform

    rank = MPI.COMM_WORLD.Get_rank()

    # atari
    if platform == 'atari':
        from bench import Monitor
        from utils.cmd import atari_arg_parser, make_atari, \
            wrap_deepmind
        from policies.nohashingcnn import CnnPolicy

        args = atari_arg_parser().parse_known_args()[0]
        if rank == 0:
            logger.configure()
        else:
            logger.configure(format_strs=[])

        workerseed = args.seed + 10000 * rank
        set_global_seeds(workerseed)
        env = make_atari(args.env)

        env = Monitor(
            env,
            logger.get_dir() and os.path.join(logger.get_dir(), str(rank))
        )
        env.seed(workerseed)

        env = wrap_deepmind(env)
        env.seed(workerseed)

        model = TRPO(CnnPolicy, env.observation_space, env.action_space)
        sess = model.single_threaded_session().__enter__()
        # model.reset_graph_and_vars()
        model.init_vars()

        fit(
            model,
            env,
            timesteps_per_batch=512,
            max_kl=0.001,
            cg_iters=10,
            cg_damping=1e-3,
            max_timesteps=int(args.num_timesteps * 1.1),
            gamma=0.98,
            lam=1.0,
            vf_iters=3,
            vf_stepsize=1e-4,
            entcoeff=0.00)
        sess.close()
        env.close()

    # mujoco
    if platform == 'mujoco':
        from policies.ppo1mlp import PPO1Mlp
        from utils.cmd import make_mujoco_env, mujoco_arg_parser
        args = mujoco_arg_parser().parse_known_args()[0]

        if rank == 0:
            logger.configure()
        else:
            logger.configure(format_strs=[])
            logger.set_level(logger.DISABLED)

        workerseed = args.seed + 10000 * rank

        env = make_mujoco_env(args.env, workerseed)

        def policy(name, observation_space, action_space):
            return PPO1Mlp(
                name,
                env.observation_space,
                env.action_space,
                hid_size=32,
                num_hid_layers=2
                )

        model = TRPO(
            policy,
            env.observation_space,
            env.action_space
        )
        sess = model.single_threaded_session().__enter__()
        model.init_vars()

        fit(
            model,
            env,
            timesteps_per_batch=1024,
            max_kl=0.01,
            cg_iters=10,
            cg_damping=0.1,
            max_timesteps=args.num_timesteps,
            gamma=0.99,
            lam=0.98,
            vf_iters=5,
            vf_stepsize=1e-3)
        sess.close()
        env.close()


if __name__ == '__main__':
    main()

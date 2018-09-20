from collections import defaultdict
from mpi4py import MPI
import os, numpy as np
import platform
import shutil
import subprocess

def sync_from_root(sess, variables, comm=None):
    """
    Send the root node's parameters to every worker.
    Arguments:
      sess: the TensorFlow session.
      variables: all parameter variables including optimizer's
    """
    if comm is None: comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    for var in variables:
        if rank == 0:
            comm.Bcast(sess.run(var))
        else:
            import tensorflow as tf
            returned_var = np.empty(var.shape, dtype='float32')
            comm.Bcast(returned_var)
            sess.run(tf.assign(var, returned_var))

def gpu_count():
    """
    Count the GPUs on this machine.
    """
    if shutil.which('nvidia-smi') is None:
        return 0
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv'])
    return max(0, len(output.split(b'\n')) - 2)

def setup_mpi_gpus():
    """
    Set CUDA_VISIBLE_DEVICES using MPI.
    """
    num_gpus = gpu_count()
    if num_gpus == 0:
        return
    local_rank, _ = get_local_rank_size(MPI.COMM_WORLD)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank % num_gpus)

def get_local_rank_size(comm):
    """
    Returns the rank of each process on its machine
    The processes on a given machine will be assigned ranks
        0, 1, 2, ..., N-1,
    where N is the number of processes on this machine.

    Useful if you want to assign one gpu per machine
    """
    this_node = platform.node()
    ranks_nodes = comm.allgather((comm.Get_rank(), this_node))
    node2rankssofar = defaultdict(int)
    local_rank = None
    for (rank, node) in ranks_nodes:
        if rank == comm.Get_rank():
            local_rank = node2rankssofar[node]
        node2rankssofar[node] += 1
    assert local_rank is not None
    return local_rank, node2rankssofar[this_node]

def share_file(comm, path):
    """
    Copies the file from rank 0 to all other ranks
    Puts it in the same place on all machines
    """
    localrank, _ = get_local_rank_size(comm)
    if comm.Get_rank() == 0:
        with open(path, 'rb') as fh:
            data = fh.read()
        comm.bcast(data)
    else:
        data = comm.bcast(None)
        if localrank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as fh:
                fh.write(data)
    comm.Barrier()

def dict_gather(comm, d, op='mean', assert_all_have_data=True):
    if comm is None: return d
    alldicts = comm.allgather(d)
    size = comm.size
    k2li = defaultdict(list)
    for d in alldicts:
        for (k,v) in d.items():
            k2li[k].append(v)
    result = {}
    for (k,li) in k2li.items():
        if assert_all_have_data:
            assert len(li)==size, "only %i out of %i MPI workers have sent '%s'" % (len(li), size, k)
        if op=='mean':
            result[k] = np.mean(li, axis=0)
        elif op=='sum':
            result[k] = np.sum(li, axis=0)
        else:
            assert 0, op
    return result

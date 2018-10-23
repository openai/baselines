# Registry of algorithms that keeps track of algorithms supported environments and 
# and fine-grained defaults for different kinds of environments (atari, retro, mujoco etc)
#
# Example usage:
# 
#   from baselines import registry
#
#   @registry.register('fancy_algorithm', supports_vecenv=False)
#   def learn(env, network):
#       return
#
#   for algo_name, algo_entry in registry.registry.items():
#       if not algo_entry['supports_vecenv']:
#           print(f'{algo_name} does not support vecenvs')
#           # should print "fancy_algorithm does not support vecenvs" (among other ones)"from baselines import logger



registry = {}

def register(name, supports_vecenv=True, defaults={}):
    def get_fn_entrypoint(fn):
        import inspect
        return '.'.join([inspect.getmodule(fn).__name__, fn.__name__])

    def _thunk(learn_fn):
        old_entry = registry.get(name)
        if old_entry is not None:
            logger.warn('Re-registering learn function {} (old entrypoint {}, new entrypoint {}) '.format(
                name, get_fn_entrypoint(old_entry['fn']), get_fn_entrypoint(learn_fn)))

        registry[name] = dict(
            fn = learn_fn,
            supports_vecenv=supports_vecenv,
            defaults=defaults,
        )
        return learn_fn
    return _thunk

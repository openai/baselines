from baselines import logger
registry = {}

def register(name, supports_vecenv=True, defaults={}, **kwargs):
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
            **kwargs
        )
        return learn_fn
    return _thunk

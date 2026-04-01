import sys


def ensure_pickle_compat() -> None:
    """Register NumPy legacy module aliases used by older SB3 checkpoints."""
    import numpy.core as np_core
    import numpy.core.multiarray as np_multiarray
    import numpy.core.numeric as np_numeric
    import numpy.random._pickle as np_pickle

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)

    for name, cls in list(np_pickle.BitGenerators.items()):
        np_pickle.BitGenerators.setdefault(cls, cls)

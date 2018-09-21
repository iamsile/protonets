def set_seed(seed: int):
    """Initialize random generators seed."""
    import tensorflow as tf
    import numpy as np
    np.random.seed(seed)
    tf.set_random_seed(seed)

def set_seed(seed: int):
    """Inititialize random generators seed."""
    import torch
    import numpy as np
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

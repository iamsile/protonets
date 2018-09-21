import os
import yaml


class Config(object):
    """Script configuration file parser.

    Attributes
    ----------
    dataset: str
        Name of the dataset to train on (i.e., 'omniglot').
    num_epochs: int
        Number of training epochs.
    learning_rate: float, optional
        Optimizer learning rate.
    num_episodes: int
        Number of episodes per epoch.
    classes_per_it_tr: int
        Number of random classes per episode for training.
    num_support_tr: int
        Number of samples per class to use as support for training.
    num_query_tr: int
        Number of samples per class to use as query for training.
    classes_per_it_val: int
        Number of random classes per episode for validation.
    num_support_val: int
        Number of samples per class to use as support for validation.
    num_query_val: int
        Number of samples per class to use as query for validation.
    seed: int
        Random seed.
    """

    def __init__(self, config_yaml: str) -> None:
        if not os.path.exists(config_yaml):
            raise ValueError(
                f"The config file at {config_yaml} is missing.")
        config = yaml.load(open(config_yaml, "r"))
        self.dataset = config["dataset"]
        self.num_epochs = config.get("num_epochs", 100)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.num_episodes = config.get("num_episodes", 100)
        self.classes_per_it_tr = config.get("classes_per_it_tr", 60)
        self.num_support_tr = config.get("num_support_tr", 5)
        self.num_query_tr = config.get("num_query_tr", 5)
        self.classes_per_it_val = config.get("classes_per_it_val", 5)
        self.num_support_val = config.get("num_support_val", 5)
        self.num_query_val = config.get("num_query_val", 15)
        self.seed = config.get("seed", 0)

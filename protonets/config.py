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
    num_episodes: int
        Number of episodes per epoch.
    num_ways_train: int
        Number of random classes per episode for training.
    num_support_train: int
        Number of samples per class to use as support for training.
    num_query_train: int
        Number of samples per class to use as query for training.
    num_ways_val: int
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
        self.num_episodes = config.get("num_episodes", 100)
        self.num_ways_train = config.get("num_ways_train", 60)
        self.num_support_train = config.get("num_support_train", 5)
        self.num_query_train = config.get("num_query_train", 5)
        self.num_ways_val = config.get("num_ways_val", 5)
        self.num_support_val = config.get("num_support_val", 5)
        self.num_query_val = config.get("num_query_val", 15)
        self.seed = config.get("seed", 0)

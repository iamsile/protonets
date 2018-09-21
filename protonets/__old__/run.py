import torch.optim as optim
from protonets.config import Config
from protonets.utils.seed import set_seed
from protonets.model.net import Protonet
from protonets.model.trainer import Trainer


def run(config: Config):
    """Entry point, experiment runner."""
    # initialize random generators seed
    set_seed(config.seed)
    # initialize data loaders
    if config.dataset == "omniglot":
        from protonets.data.omniglot import OmniglotDataLoader
        train_loader = OmniglotDataLoader(
            "train", config.classes_per_it_tr, config.num_episodes)
        # val_loader = OmniglotDataLoader(
        #     "val", config.classes_per_it_val, config.num_episodes)
    else:
        raise ValueError(f"Unrecognized dataset {config.dataset} provided.")
    # initialize model
    model = Protonet()
    # initialize optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=config.learning_rate)
    # initialize trainer
    trainer = Trainer(train_loader, config.num_epochs,
                      config.num_support_tr, None)
    # model training
    results = trainer(model, optimizer)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = results
    print(best_state, best_acc, train_loss, train_acc, val_loss, val_acc)

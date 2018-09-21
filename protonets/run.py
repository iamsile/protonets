from protonets.config import Config
from protonets.utils.seed import set_seed
from protonets.model.net import Protonet
from tqdm import trange
import matplotlib.pyplot as plt
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def run(config: Config):
    """Entry point, experiment runner."""
    # initialize random generators seed
    set_seed(config.seed)
    # initialize data loaders
    if config.dataset == "omniglot":
        from protonets.data.omniglot import OmniglotDataSet
        dataset_train = OmniglotDataSet("train")
        dataset_val = OmniglotDataSet("val")
        dataset_test = OmniglotDataSet("test")
    else:
        raise ValueError(f"Unrecognized dataset {config.dataset} provided.")
    # initialize model
    model = Protonet()
    # enable interactive plotting
    plt.ion()
    fig, axes = plt.subplots(figsize=(20., 8.), ncols=2)
    axes[0].set(title="Prototypical Networks Loss",
                xlabel="Number of Epochs", ylabel="Error")
    axes[1].set(title="Prototypical Networks Accuracy",
                xlabel="Number of Epochs", ylabel="Accuracy")
    for ax in axes:
        ax.grid(which="minor", linestyle=":")
    # training loop
    step = 1. / config.num_episodes
    for epoch in trange(config.num_epochs, leave=True, position=1):
        for episode in trange(config.num_episodes, leave=True, position=2):
            # sample training data
            support_train, query_train, labels_train = dataset_train.sample(
                config.num_episodes, config.num_support_train, config.num_query_train)
            # in-sample performance metrics
            loss_train, accuracy_train = model.train(
                support_train, query_train, labels_train)
            # visualization
            if epoch == episode == 0:
                axes[0].scatter(epoch+step*episode, loss_train,
                                label="training", color=COLORS[0])
                axes[1].scatter(epoch+step*episode,
                                accuracy_train, label="training", color=COLORS[0])
            else:
                axes[0].scatter(epoch+step*episode,
                                loss_train, color=COLORS[0])
                axes[1].scatter(epoch+step*episode,
                                accuracy_train, color=COLORS[0])
            plt.draw()
            plt.pause(0.05)
        # sample validation data
        support_val, query_val, labels_val = dataset_val.sample(
            config.num_episodes, config.num_support_val, config.num_query_val)
        # out-of sample performance
        loss_val, accuracy_val = model.evaluate(
            support_val, query_val, labels_val)
        # visualization
        if epoch == 0:
            axes[0].scatter(epoch+1, loss_val,
                            label="validation", color=COLORS[1])
            axes[1].scatter(epoch+1, accuracy_val,
                            label="validation", color=COLORS[1])
            for ax in axes:
                ax.legend()
        else:
            axes[0].scatter(epoch+1, loss_val, color=COLORS[1])
            axes[1].scatter(epoch+1, accuracy_val, color=COLORS[1])
        plt.draw()
        plt.pause(0.05)
    fig.savefig(f"assets/{config.dataset}.pdf", format="pdf", dpi=300)
    # disable interactive plotting
    plt.ioff()
    # sample testing data
    support_test, query_test, labels_test = dataset_test.sample(
        config.num_episodes, config.num_support_val, config.num_query_val)
    # out-of sample performance
    loss_test, accuracy_test = model.evaluate(
        support_test, query_test, labels_test)
    # overall performance out-of sample
    print(f"Loss: {loss_test:.5}, Accuracy: {accuracy_test:.5}")


if __name__ == "__main__":
    import sys
    run(Config(f"config/{sys.argv[1]}.yaml"))

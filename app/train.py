from protonets.run import run
from protonets.config import Config


def main(config_path: str):
    """Entry point."""
    config = Config(config_path)
    run(config)


if __name__ == "__main__":
    main("config/omniglot.yaml")

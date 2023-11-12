import argparse

parser = argparse.ArgumentParser(
    description="Reinforcement Learning: Lunar Lander",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--steps", type=int, default=250,
    help="Maximum number of steps"
)

parser.add_argument(
    "--device", type=str, default="cpu",
    choices=["cpu", "cuda"],
    help="Device to run"
)

subparsers = parser.add_subparsers(
    dest="command",
    required=True
)

train_parser = subparsers.add_parser(
    "train",
    help="Train an agent to play the game"
)

train_parser.add_argument(
    "--envs", type=int, default=30,
    help="Number of parallel environments to run"
)

train_parser.add_argument(
    "--epochs", type=int, default=500,
    help="Number of training epochs"
)


train_parser.add_argument(
    "--randomize", action=argparse.BooleanOptionalAction,
    help="Randomize environments"
)

train_parser.add_argument(
    "--save-interval", type=int, default=200,
    help="Interval (in epochs) to save the models"
)

train_parser.add_argument(
    "--load", type=str, required=False,
    help="Load agent from a saved state"
)


play_parser = subparsers.add_parser(
    "play",
    help="Play game from a saved state"
)

play_parser.add_argument(
    "path", type=str,
    help="Path of the saved model"
)

args = parser.parse_args()

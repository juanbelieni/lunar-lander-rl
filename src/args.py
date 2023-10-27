import argparse

parser = argparse.ArgumentParser(
    description="Reinforcement Learning: Lunar Lander",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--envs", type=int, default=30,
    help="Number of parallel environments to run"
)

parser.add_argument(
    "--epochs", type=int, default=200,
    help="Number of training epochs"
)

parser.add_argument(
    "--steps", type=int, default=300,
    help="Number of steps per epoch"
)

parser.add_argument(
    "--device", type=str, default="cpu",
    choices=["cpu", "cuda"],
    help="Device to run"
)

args = parser.parse_args()

from common.imports import *
from common.utils import str2bool

def get_alg_args() -> Namespace:
    """Parse command line arguments for the conditioned PPO variant."""
    parser = ap.ArgumentParser()

    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="Total timesteps for the experiment")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Number of steps per policy rollout")
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Timesteps between evaluations")

    parser.add_argument("--lr-actor", type=float, default=3e-4,
                        help="Learning rate for the actor")
    parser.add_argument("--lr-critic", type=float, default=3e-4,
                        help="Learning rate for the critic")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="Lambda for GAE")
    parser.add_argument("--ppo-epochs", type=int, default=10,
                        help="Number of PPO update epochs")
    parser.add_argument("--clip-ratio", type=float, default=0.2,
                        help="Clipping ratio for the surrogate objective")

    parser.add_argument("--cvi", type=str2bool, default=False,
                        help="Use conditioned value input")
    parser.add_argument("--vve", type=str2bool, default=False,
                        help="Use value function with threshold input")
    parser.add_argument("--threshold-encoding", type=str, default="raw",
                        help="Encoding type for thresholds")
    parser.add_argument("--threshold-embed-dim", type=int, default=16,
                        help="Embedding dimension for threshold encoding")
    parser.add_argument("--dynamic-threshold", type=str2bool, default=False,
                        help="Sample thresholds dynamically during training")
    parser.add_argument("--threshold-min", type=float, default=0.0,
                        help="Minimum threshold value when dynamic")
    parser.add_argument("--threshold-max", type=float, default=1.0,
                        help="Maximum threshold value when dynamic")
    parser.add_argument("--fixed-threshold", type=float, default=0.0,
                        help="Fixed threshold when not dynamic")

    return parser.parse_known_args()[0]
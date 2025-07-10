from grid2op.gym_compat import BoxGymActSpace, DiscreteActSpace # if we import gymnasium, GymEnv will convert to Gymnasium!   

from alg.dqn.core import DQN
from alg.lagr_ppo.core import LagrPPO
from alg.ppo.core import PPO
from alg.sac.core import SAC
from alg.td3.core import TD3
from common.checkpoint import (
    DQNCheckpoint,
    LagrPPOCheckpoint,
    PPOCheckpoint,
    SACCheckpoint,
    TD3Checkpoint,
    CPPOCheckpoint,
)
from common.imports import *
from common.utils import set_random_seed, set_torch, str2bool
from env.config import get_env_args
from env.utils import auxiliary_make_env
from alg.ppo_cppo.core import CPPO

# Dictionary mapping algorithm names to their corresponding classes
ALGORITHMS: Dict[str, Type[Any]] = {
    'DQN': DQN,
    'PPO': PPO,
    'SAC': SAC,
    'TD3': TD3,
    'LAGRPPO': LagrPPO,
    'PPO_CCPO': CPPO,
}

def main(args: Namespace) -> None:
    """
    Main function to run the RL algorithms based on the provided arguments.

    Args:
        args (Namespace): Command line arguments parsed by argparse.

    Raises:
        AssertionError: If time limit exceeds 2800 minutes or if number of environments is less than 1.
        AssertionError: If the specified algorithm is not supported.
    """
    assert args.time_limit <= 2800, f"Invalid time limit: {args.time_limit}. Timeout limit is : 2800"
    start_time = time()
    
    # Update args with environment arguments
    args = ap.Namespace(**vars(args), **vars(get_env_args()))
    assert args.n_envs >= 1, f"Invalid n° of environments: {args.n_envs}. Must be >= 1"
    
    alg = args.alg.upper()
    assert alg in ALGORITHMS.keys(), f"Unsupported algorithm: {alg}. Supported algorithms are: {ALGORITHMS}"
    if (alg == "LAGRPPO" and args.constraints_type == 0) or (alg != "LAGRPPO" and args.constraints_type in [1, 2]):
        raise ValueError("Check the constrained version of the alg/env!")

    # run_name = args.resume_run_name if args.resume_run_name \
    #     else f"{args.alg}_{args.env_id}_{"T" if args.action_type == "topology" else "R"}_{args.seed}_{args.difficulty}_{"H" if args.use_heuristic else ""}_{"I" if args.heuristic_type == "idle" else ""}_{"C1" if args.constraints_type == 1 else "C2" if args.constraints_type == 2 else ""}_{int(time())}_{np.random.randint(0, 50000)}"
    run_name = args.resume_run_name if args.resume_run_name else (
        f"{args.alg}_{args.env_id}_"
        f"{'T' if args.action_type == 'topology' else 'R'}_"
        f"{args.seed}_{args.difficulty}_"
        f"{'H' if args.use_heuristic else ''}_"
        f"{'I' if args.heuristic_type == 'idle' else ''}_"
        f"{'C1' if args.constraints_type == 1 else 'C2' if args.constraints_type == 2 else ''}_"
        f"{int(time())}_{np.random.randint(0, 50000)}"
)

    # Initialize the appropriate checkpoint based on the algorithm
    if alg == 'LAGRPPO':
        checkpoint = LagrPPOCheckpoint(run_name, args)
    elif alg == 'DQN':
        checkpoint = DQNCheckpoint(run_name, args)
    elif alg == 'PPO':
        checkpoint = PPOCheckpoint(run_name, args)
    elif alg == 'PPO_CCPO':
        checkpoint = CPPOCheckpoint(run_name, args)
    elif alg == 'SAC':
        checkpoint = SACCheckpoint(run_name, args)
    elif alg == 'TD3':
        checkpoint = TD3Checkpoint(run_name, args)
    else:
        pass  # This case should not occur due to earlier assertion

    # Set random seed and Torch configuration
    set_random_seed(args.seed)
    set_torch(args.n_threads, args.th_deterministic, args.cuda)
    
    # Resume run if checkpoint was resumed
    if checkpoint.resumed: args = checkpoint.loaded_run['args']

    # Create multiple async environments for parallel processing
    main_gym_env, main_g2o_env = auxiliary_make_env(args, checkpoint.resumed)

    with Manager() as manager:
        if args.action_type == "topology":
            print("Initializing the distributed action 'mapper'... (takes a while with big action spaces)")
            shared_action_space = manager.list(main_gym_env.action_space.converter.all_actions)

        def make_vec_subprocess(idx):
            if args.action_type == "topology":
                action_space = DiscreteActSpace(main_g2o_env.action_space,
                                                    action_list=shared_action_space)
                return auxiliary_make_env(args, resume_run=checkpoint.resumed, idx=idx, action_space=action_space)[0]
            
            return auxiliary_make_env(args, resume_run=checkpoint.resumed, idx=idx)[0]
from .agent import ConditionedPPO
from .config import get_alg_args
from common.checkpoint import CheckpointSaver
from common.imports import *

class CPPO:
    """Thin wrapper running the ConditionedPPO agent."""

    def __init__(self, envs: gym.Env, run_name: str, start_time: float, args: Dict[str, Any], ckpt: CheckpointSaver):
        if not ckpt.resumed:
            args = ap.Namespace(**vars(args), **vars(get_alg_args()))
        self.args = args
        self.agent = ConditionedPPO(envs, args)
        self.ckpt = ckpt
        self.envs = envs

        if ckpt.resumed:
            self.agent.load_state_dict(ckpt.loaded_run['agent'])

        self.run_name = run_name
        self.start_time = start_time
        self._train()

    def _train(self) -> None:
        try:
            self.agent.train_loop(self.args.total_timesteps)
        finally:
            self.ckpt.set_record(self.args, self.agent.state_dict(), 0, "", 0)
            self.ckpt.save()
            self.envs.close()
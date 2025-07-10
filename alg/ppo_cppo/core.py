from time import time

from .agent import ConditionedPPO
from .config import get_alg_args
from common.checkpoint import CPPOCheckpoint
from common.imports import *
from common.logger import Logger
from env.eval import Evaluator

class CPPO:
    """Thin wrapper running the ConditionedPPO agent."""

    def __init__(self, envs: gym.Env, run_name: str, start_time: float, args: Dict[str, Any], ckpt: CPPOCheckpoint):
        if not ckpt.resumed:
            args = ap.Namespace(**vars(args), **vars(get_alg_args()))
        self.args = args
        self.agent = ConditionedPPO(envs, args)
        self.ckpt = ckpt
        self.envs = envs

        if ckpt.resumed:
            self.agent.load_state_dict(ckpt.loaded_run['agent'])

        assert args.eval_freq % args.n_envs == 0, \
            f"Invalid eval frequency: {args.eval_freq}. Must be multiple of n_envs {args.n_envs}"

        self.logger = Logger(run_name, args) if args.track else None
        self.evaluator = Evaluator(args, self.logger, self.agent.device)

        self.run_name = run_name
        self.start_time = start_time
        self._train()

    def _train(self) -> None:
        global_step = 0 if not self.ckpt.resumed else self.ckpt.loaded_run.get('global_step', 0)
        try:
            self.agent.train(
                self.args.total_timesteps,
                logger=self.logger,
                evaluator=self.evaluator,
                start_time=self.start_time,
            )
        finally:
            self.ckpt.set_record(
                self.args,
                self.agent.state_dict(),
                global_step,
                "" if not self.logger else self.logger.wb_path,
                global_step,
            )
            self.ckpt.save()
            if self.logger:
                self.logger.close()
            self.envs.close()
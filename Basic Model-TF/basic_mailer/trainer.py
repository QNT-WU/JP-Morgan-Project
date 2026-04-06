"""Backward-compatible training entrypoints.

The package now exposes class-based trainers under :mod:`basic_mailer.training`,
but these functional wrappers remain available for existing scripts and tests.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import ModelParams, NetParams, Obj3Params, TrainParams
from .io_utils import JSONLLogger, TFCheckpointIO
from .networks import PolicyNet, ValueNet
from .training import Objective1Trainer, Objective2Trainer, Objective3Trainer


def train_objective_1(
    mp: ModelParams,
    npol: NetParams,
    tp: TrainParams,
    jsonl_logger: Optional[JSONLLogger] = None,
    ckptio: Optional[TFCheckpointIO] = None,
) -> Tuple[PolicyNet, Dict[str, List[float]]]:
    """Train Objective 1 using the class-based trainer."""
    trainer = Objective1Trainer(
        mp=mp,
        tp=tp,
        npol=npol,
        jsonl_logger=jsonl_logger,
        ckptio=ckptio,
    )
    return trainer.train()


def train_objective_2(
    mp: ModelParams,
    npol: NetParams,
    tp: TrainParams,
    jsonl_logger: Optional[JSONLLogger] = None,
    ckptio: Optional[TFCheckpointIO] = None,
) -> Tuple[PolicyNet, Dict[str, List[float]]]:
    """Train Objective 2 using the class-based trainer."""
    trainer = Objective2Trainer(
        mp=mp,
        tp=tp,
        npol=npol,
        jsonl_logger=jsonl_logger,
        ckptio=ckptio,
    )
    return trainer.train()


def train_objective_3(
    mp: ModelParams,
    npol: NetParams,
    nval: NetParams,
    tp: TrainParams,
    op3: Obj3Params,
    jsonl_logger: Optional[JSONLLogger] = None,
    ckptio: Optional[TFCheckpointIO] = None,
) -> Tuple[PolicyNet, ValueNet, Dict[str, List[float]]]:
    """Train Objective 3 using the class-based trainer."""
    trainer = Objective3Trainer(
        mp=mp,
        tp=tp,
        op3=op3,
        npol=npol,
        nval=nval,
        jsonl_logger=jsonl_logger,
        ckptio=ckptio,
    )
    return trainer.train()

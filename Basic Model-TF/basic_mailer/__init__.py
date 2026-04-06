"""Client-facing package for the basic ``(k, z)`` model in TensorFlow.

The package exposes structural configuration objects, TensorFlow models,
training utilities, and IO helpers while keeping backward compatibility with
legacy module paths used by the original research prototype.
"""

from .config import ModelParams, NetParams, Obj3Params, TrainParams
from .io_utils import JSONLLogger, TFCheckpointIO
from .networks import PolicyModel, PolicyNet, ValueModel, ValueNet
from .pipeline import BasicMailerPipeline, PipelineArgs
from .training import BaseTrainer, Objective1Trainer, Objective2Trainer, Objective3Trainer

__all__ = [
    "BasicMailerPipeline",
    "BaseTrainer",
    "JSONLLogger",
    "PipelineArgs",
    "ModelParams",
    "NetParams",
    "Obj3Params",
    "Objective1Trainer",
    "Objective2Trainer",
    "Objective3Trainer",
    "PolicyModel",
    "PolicyNet",
    "TFCheckpointIO",
    "TrainParams",
    "ValueModel",
    "ValueNet",
]

"""Training API exposed under a dedicated package namespace."""

from risky_debt.trainer import (
    BaseObjectiveTrainer,
    Objective1Trainer,
    Objective2Trainer,
    Objective3Trainer,
    ObjectiveTrainingArtifacts,
    train_objective_1,
    train_objective_2,
    train_objective_3,
)

__all__ = [
    "BaseObjectiveTrainer",
    "Objective1Trainer",
    "Objective2Trainer",
    "Objective3Trainer",
    "ObjectiveTrainingArtifacts",
    "train_objective_1",
    "train_objective_2",
    "train_objective_3",
]

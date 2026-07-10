"""Student training and distillation methods."""

from learning_to_rank_distillation.distillation.feature_based import train_feature_based_student
from learning_to_rank_distillation.distillation.no_kd_baseline import train_no_kd_student
from learning_to_rank_distillation.distillation.relation_based import train_relation_based_student
from learning_to_rank_distillation.distillation.response_based import train_response_based_student

__all__ = [
    "train_feature_based_student",
    "train_no_kd_student",
    "train_relation_based_student",
    "train_response_based_student",
]


import torch
from torch import Tensor
from torch.nn import functional as F


class DistillationLossOutput:
    total_loss:Tensor
    hard_loss:Tensor
    distillation_loss:Tensor
    scaled_distillation_loss:Tensor

def calculate_response_distillation_loss(
    student_logits:Tensor,
    teacher_logits:Tensor,
    labels:Tensor,
    temperature:float,
    hard_loss_weight:float,
)-> tuple[Tensor,Tensor,Tensor]:
    if temperature <=0:
        raise ValueError("temperature must be greater than 0")
    
    if not 0.0 <= hard_loss_weight <=1.0:
        raise ValueError(
            "hard_loss_weight must be between 0 and 1"
        )
    
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have "
            f"the same shape, but got "
            f"{student_logits.shape} and {teacher_logits.shape}"
        )
    
    hard_loss = F.cross_entropy(
        student_logits,
        labels,
    )
    teacher_probabilities = F.softmax(
        teacher_logits.detach()/temperature,
        dim = 1
    )
    student_log_probabilities = F.log_softmax(
        student_logits / temperature,
        dim = 1,
    )
    kd_loss = F.kl_div(
        input = student_log_probabilities,
        target = teacher_probabilities,
        reduction = "batchmean",
    )
    kd_loss = temperature**2 * kd_loss

    total_loss = (
        hard_loss_weight*hard_loss + (1-hard_loss_weight)* kd_loss
    )
    return total_loss, hard_loss, kd_loss
    

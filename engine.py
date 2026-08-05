import torch 
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from losses import calculate_response_distillation_loss

def train_supervised_one_epoch(
    model:nn.Module,
    data_loader:DataLoader,
    optimizer:Optimizer,
    device:torch.device,
):
    model.train()
    accumulated_loss = 0
    number_of_correct_predictions = 0
    number_of_samples = 0
    for image,labels in data_loader:
        images = image.to(
            device,
            non_blocking = True,
        )
        labels = labels.to(
            device,
            non_blocking = True,
        )
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = nn.functional.cross_entropy(
            logits,
            labels,
        )
        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        accumulated_loss += loss.item()*batch_size
        
        number_of_correct_predictions +=(
            logits.argmax(dim=1).eq(labels).sum().item()
        )
        number_of_samples += batch_size
    average_loss = (
        accumulated_loss / number_of_samples
    )
    accuracy = (
        number_of_correct_predictions
        / number_of_samples
    )
    return average_loss,accuracy

def train_kd_one_epoch(
    student:nn.Module,
    teacher:nn.Module,
    data_loader:DataLoader,
    optimizer:Optimizer,
    device:torch.device,
    temperature:float,
    hard_loss_weight:float,
    epoch_index:int,
)->tuple[float,float,float,float]:
    student.train()
    teacher.eval()

    accumulated_total_loss = 0.0
    accumulated_hard_loss = 0.0
    accumulated_kd_loss = 0.0

    number_of_correct_predictions = 0
    number_of_samples = 0

    for batch_index,(images,labels) in enumerate(
        data_loader
    ):
        images = images.to(
            device,
            non_blocking = True,
        )
        labels = labels.to(
            device,
            non_blocking =True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_logits = teacher(images)
        student_logits = student(images)
        total_loss, hard_loss, kd_loss = (
            calculate_response_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=temperature,
                hard_loss_weight=hard_loss_weight,
            )
        )

        if epoch_index ==0 and batch_index == 0:
            print("\nFirst KD batch check:")
            print(
                "  teacher logits shape:",
                teacher_logits.shape,
            )
            print(
                "  student logits shape:",
                student_logits.shape,
            )
            print(
                "  hard loss:",
                f"{hard_loss.item():.4f}",
            )
            print(
                "  scaled KD loss:",
                f"{kd_loss.item():.4f}",
            )
            print()
        total_loss.backward()
        optimizer.step()
        batch_size = labels.size(0)

        accumulated_total_loss+=(
            total_loss.item() * batch_size
        )
        accumulated_hard_loss += (
            hard_loss.item() * batch_size
        )
        accumulated_kd_loss += (
            kd_loss.item() * batch_size
        )
        number_of_correct_predictions +=(
            student_logits.argmax(dim=1).eq(labels).sum().item()
        )
        number_of_samples += batch_size
    return (
        accumulated_total_loss / number_of_samples,
        accumulated_hard_loss / number_of_samples,
        accumulated_kd_loss / number_of_samples,
        number_of_correct_predictions / number_of_samples,
    )

@torch.no_grad()
def evaluate(
    model:nn.Module,
    data_loader:DataLoader,
    device:torch.device,
):
    model.eval()

    accumulated_loss = 0.0
    number_of_correct_predictions = 0
    number_of_samples = 0

    for images, labels in data_loader:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        loss = nn.functional.cross_entropy(
            logits,
            labels,
        )
        batch_size = labels.size(0)

        accumulated_loss += loss.item() * batch_size
        number_of_correct_predictions += (
            logits.argmax(dim=1)
            .eq(labels)
            .sum()
            .item()
        )
        number_of_samples += batch_size

    average_loss = (
        accumulated_loss / number_of_samples
    )
    accuracy = (
        number_of_correct_predictions
        / number_of_samples
    )
    return average_loss, accuracy

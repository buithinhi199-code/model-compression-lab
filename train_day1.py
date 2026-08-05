import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import TrainingConfig
from data import build_data_loaders
from engine import (
    evaluate,
    train_kd_one_epoch,
    train_supervised_one_epoch,
)
from models import (
    StudentCNN,
    TeacherCNN,
    count_trainable_parameters,
)

def set_random_seed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(
    model:nn.Module,
    path:Path,
    epoch:int,
    validation_accuracy:float,
):
    torch.save(
        {
            "model_state_dict":model.state_dict(),
            "epoch":epoch,
            "validation_accuracy":validation_accuracy,
        },
        path,
    )
def load_checkpoint(
    model:nn.Module,
    path:Path,
    device:torch.device,
):
    checkpoint = torch.load(
        path,
        map_location = device,
        weights_only = True,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
def train_supervised_model(
    model_name: str,
    model: nn.Module,
    training_loader,
    validation_loader,
    device: torch.device,
    config: TrainingConfig,
    number_of_epochs: int,
    checkpoint_path: Path,
) -> None:
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=number_of_epochs,
        eta_min=config.minimum_learning_rate,
    )

    best_validation_accuracy = -1.0

    print(f"\n===== Training {model_name} =====")

    for epoch in range(number_of_epochs):
        training_loss, training_accuracy = (
            train_supervised_one_epoch(
                model=model,
                data_loader=training_loader,
                optimizer=optimizer,
                device=device,
            )
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_loader,
            device=device,
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )
        print(
            f"{model_name} | "
            f"epoch={epoch + 1:02d}/{number_of_epochs} | "
            f"lr={current_learning_rate:.6f} | "
            f"train_loss={training_loss:.4f} | "
            f"train_acc={training_accuracy:.2%} | "
            f"val_loss={validation_loss:.4f} | "
            f"val_acc={validation_accuracy:.2%}"
        )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_accuracy
            )

            save_checkpoint(
                model=model,
                path=checkpoint_path,
                epoch=epoch + 1,
                validation_accuracy=validation_accuracy,
            )

        scheduler.step()
    load_checkpoint(
        model=model,
        path=checkpoint_path,
        device=device,
    )
def train_distilled_student(
    student: nn.Module,
    teacher: nn.Module,
    training_loader,
    validation_loader,
    device: torch.device,
    config: TrainingConfig,
    checkpoint_path: Path,
) -> None:
    teacher.eval()
    teacher.requires_grad_(False)

    optimizer = AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.kd_epochs,
        eta_min=config.minimum_learning_rate,
    )

    best_validation_accuracy = -1.0

    print("\n===== Training Response KD Student =====")

    for epoch in range(config.kd_epochs):
        (
            total_loss,
            hard_loss,
            kd_loss,
            training_accuracy,
        ) = train_kd_one_epoch(
            student=student,
            teacher=teacher,
            data_loader=training_loader,
            optimizer=optimizer,
            device=device,
            temperature=config.temperature,
            hard_loss_weight=config.hard_loss_weight,
            epoch_index=epoch,
        )

        validation_loss, validation_accuracy = evaluate(
            model=student,
            data_loader=validation_loader,
            device=device,
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            "Response KD Student | "
            f"epoch={epoch + 1:02d}/{config.kd_epochs} | "
            f"lr={current_learning_rate:.6f} | "
            f"total_loss={total_loss:.4f} | "
            f"hard_loss={hard_loss:.4f} | "
            f"kd_loss={kd_loss:.4f} | "
            f"train_acc={training_accuracy:.2%} | "
            f"val_acc={validation_accuracy:.2%}"
        )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_accuracy
            )

            save_checkpoint(
                model=student,
                path=checkpoint_path,
                epoch=epoch + 1,
                validation_accuracy=validation_accuracy,
            )

        scheduler.step()

    load_checkpoint(
        model=student,
        path=checkpoint_path,
        device=device,
    )
def main() -> None:
    config = TrainingConfig()

    checkpoint_directory = Path(
        config.checkpoint_directory
    )
    results_directory = Path(
        config.results_directory
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )
    # --------------------------------------------------
    # 1. Teacher baseline
    # --------------------------------------------------
    set_random_seed(config.teacher_seed)

    (
        teacher_training_loader,
        teacher_validation_loader,
        teacher_test_loader,
    )= build_data_loaders(
        config = config,
        shuffle_seed = config.teacher_seed,
    )
    teacher = TeacherCNN().to(device)

    print(
        "Teacher parameters:",
        f"{count_trainable_parameters(teacher):,}",
    )

    train_supervised_model(
        model_name = "Teacher",
        model = teacher,
        training_loader = teacher_training_loader,
        validation_loader = teacher_validation_loader,
        device = device,
        config = config,
        number_of_epochs = config.teacher_epochs,
        checkpoint_path = checkpoint_directory / "teacher_best.pt",
    )

    teacher_test_loss,teacher_test_accuracy = evaluate(
        model = teacher,
        data_loader = teacher_test_loader,
        device = device,
    )
    del teacher_training_loader, teacher_validation_loader, teacher_test_loader

    # --------------------------------------------------
    # 2. Student baseline
    # --------------------------------------------------
    set_random_seed(config.student_seed)

    (
        student_training_loader,
        student_validation_loader,
        student_test_loader,
    )= build_data_loaders(
        config = config,
        shuffle_seed = config.student_seed,
    )
    
    baseline_student = StudentCNN().to(device)
    
    print(
        "\nStudent parameters:",
        f"{count_trainable_parameters(baseline_student):,}",
    )

    train_supervised_model(
        model_name = "Baseline Student",
        model = baseline_student,
        training_loader = student_training_loader,
        validation_loader = student_validation_loader,
        device = device,
        config = config,
        number_of_epochs = config.student_epochs,
        checkpoint_path = checkpoint_directory / "baseline_student_best.pt",
    )
    
    (
        baseline_test_loss,
        baseline_test_accuracy,
    ) = evaluate(
        model = baseline_student,
        data_loader = student_test_loader,
        device = device,
    )
    del student_training_loader, student_validation_loader, student_test_loader

    # --------------------------------------------------
    # 3. Student with response KD
    # --------------------------------------------------
    set_random_seed(config.student_seed)

    (
        kd_training_loader,
        kd_validation_loader,
        kd_test_loader,
    )= build_data_loaders(
        config = config,
        shuffle_seed = config.student_seed,
    )
    kd_student = StudentCNN().to(device)
    
    train_distilled_student(
        student = kd_student,
        teacher = teacher,
        training_loader = kd_training_loader,
        validation_loader = kd_validation_loader,
        device = device,
        config = config,
        checkpoint_path = checkpoint_directory / "kd_student_best.pt",
    )

    kd_test_loss,kd_test_accuracy = evaluate(
        model = kd_student,
        data_loader = kd_test_loader,
        device = device,
    )
    
    # --------------------------------------------------
    # 4. Save summary
    # --------------------------------------------------

    summary = {
        "device": str(device),
        "temperature": config.temperature,
        "hard_loss_weight": config.hard_loss_weight,
        "teacher": {
            "parameters": count_trainable_parameters(
                teacher
            ),
            "test_loss": teacher_test_loss,
            "test_accuracy": teacher_test_accuracy,
        },
        "student_baseline": {
            "parameters": count_trainable_parameters(
                baseline_student
            ),
            "test_loss": baseline_test_loss,
            "test_accuracy": baseline_test_accuracy,
        },
        "response_kd_student": {
            "parameters": count_trainable_parameters(
                kd_student
            ),
            "test_loss": kd_test_loss,
            "test_accuracy": kd_test_accuracy,
        },
        "kd_improvement": (
            kd_test_accuracy
            - baseline_test_accuracy
        ),
    }

    summary_path = (
        results_directory / "day1_summary.json"
    )

    with summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\n===== Day 1 Final Results =====")
    print(
        f"Teacher test accuracy: "
        f"{teacher_test_accuracy:.2%}"
    )
    print(
        f"Student baseline test accuracy: "
        f"{baseline_test_accuracy:.2%}"
    )
    print(
        f"Response KD Student test accuracy: "
        f"{kd_test_accuracy:.2%}"
    )
    print(
        f"KD improvement over baseline: "
        f"{kd_test_accuracy - baseline_test_accuracy:+.2%}"
    )
    print(
        "Saved summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()


    
    

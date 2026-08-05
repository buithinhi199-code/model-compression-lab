import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from config import TrainingConfig
from data import build_data_loaders
from engine import train_kd_one_epoch
from losses import calculate_response_distillation_loss
from models import StudentCNN, TeacherCNN, count_trainable_parameters


class FakeCIFAR10(Dataset):
    def __init__(self, root, train, download, transform):
        self.length = 50_000 if train else 10_000

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.zeros(3, 32, 32), 0


class ModelTests(unittest.TestCase):
    def test_model_output_shapes_and_parameter_counts(self):
        inputs = torch.randn(2, 3, 32, 32)
        teacher = TeacherCNN()
        student = StudentCNN()

        self.assertEqual(teacher(inputs).shape, (2, 10))
        self.assertEqual(student(inputs).shape, (2, 10))
        self.assertGreater(
            count_trainable_parameters(teacher),
            count_trainable_parameters(student),
        )


class DataTests(unittest.TestCase):
    @patch("data.datasets.CIFAR10", FakeCIFAR10)
    def test_data_split_sizes(self):
        config = TrainingConfig(
            batch_size=32,
            number_of_workers=0,
            validation_size=5_000,
        )

        training_loader, validation_loader, test_loader = (
            build_data_loaders(config, shuffle_seed=123)
        )

        self.assertEqual(len(training_loader.dataset), 45_000)
        self.assertEqual(len(validation_loader.dataset), 5_000)
        self.assertEqual(len(test_loader.dataset), 10_000)


class DistillationTests(unittest.TestCase):
    def test_distillation_loss_backpropagates(self):
        student_logits = torch.randn(4, 10, requires_grad=True)
        teacher_logits = torch.randn(4, 10)
        labels = torch.tensor([0, 1, 2, 3])

        total_loss, hard_loss, kd_loss = (
            calculate_response_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=4.0,
                hard_loss_weight=0.5,
            )
        )
        total_loss.backward()

        self.assertTrue(math.isfinite(total_loss.item()))
        self.assertTrue(math.isfinite(hard_loss.item()))
        self.assertTrue(math.isfinite(kd_loss.item()))
        self.assertIsNotNone(student_logits.grad)

    def test_kd_epoch_returns_finite_metrics(self):
        inputs = torch.randn(8, 4)
        labels = torch.randint(0, 3, (8,))
        loader = DataLoader(
            TensorDataset(inputs, labels),
            batch_size=4,
        )
        student = nn.Linear(4, 3)
        teacher = nn.Linear(4, 3)
        optimizer = torch.optim.SGD(
            student.parameters(),
            lr=0.01,
        )

        metrics = train_kd_one_epoch(
            student=student,
            teacher=teacher,
            data_loader=loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            temperature=4.0,
            hard_loss_weight=0.5,
            epoch_index=1,
        )

        self.assertEqual(len(metrics), 4)
        self.assertTrue(all(math.isfinite(value) for value in metrics))


if __name__ == "__main__":
    unittest.main()

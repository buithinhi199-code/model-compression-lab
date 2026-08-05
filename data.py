import torch 
from torch.utils.data import DataLoader,Subset
from torchvision import datasets, transforms

from config import TrainingConfig

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)
CIFAR10_STANDARD_DEVIATION = (
    0.2470,
    0.2435,
    0.2616,
)

def build_training_transform()-> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(
                size=32,
                padding = 4,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = CIFAR10_MEAN,
                std = CIFAR10_STANDARD_DEVIATION,
            )
        ]
    )

def build_evaluation_transform()->transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean = CIFAR10_MEAN,
                std = CIFAR10_STANDARD_DEVIATION,
            )
        ]
    )
def build_data_loaders(
    config:TrainingConfig,
    shuffle_seed:int,
)->tuple[DataLoader,DataLoader,DataLoader]:
    training_dataset_with_augmentation = datasets.CIFAR10(
        root = config.data_directory,
        train = True,
        download = True,
        transform = build_training_transform(),
    )

    training_dataset_without_augmentation = datasets.CIFAR10(
        root = config.data_directory,
        train = True,
        download = False,
        transform = build_evaluation_transform(),
    )
    test_dataset = datasets.CIFAR10(
        root = config.data_directory,
        train = False,
        download = True,
        transform = build_evaluation_transform(),
    )
    split_generator = torch.Generator().manual_seed(
        config.split_seed
    )
    indices = torch.randperm(
        len(training_dataset_with_augmentation),
        generator = split_generator,
    ).tolist()
    validation_indices = indices[:config.validation_size]
    training_indices = indices[config.validation_size:]
    validation_dataset = Subset(
        training_dataset_without_augmentation,
        validation_indices,
    )
    training_dataset = Subset(
        training_dataset_with_augmentation,
        training_indices,
    )
    shuffle_generator = torch.Generator().manual_seed(
        shuffle_seed
    )
    loader_arguments = {
        "batch_size":config.batch_size,
        "num_workers":config.number_of_workers,
        "pin_memory":True,
        "persistent_workers":(
            config.number_of_workers>0
        ),
    }
    training_loader = DataLoader(
        dataset = training_dataset,
        shuffle = True,
        generator = shuffle_generator,
        **loader_arguments,
    )
    validation_loader = DataLoader(
        dataset = validation_dataset,
        shuffle = False,
        **loader_arguments,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        **loader_arguments,
    )
    return (
        training_loader,validation_loader,test_loader,
    )

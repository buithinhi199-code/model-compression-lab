from dataclasses import dataclass

@dataclass(frozen = True)
class TrainingConfig:
    data_directory:str = "data"
    checkpoint_directory:str = "checkpoints"
    results_directory:str = "results"

    batch_size:int = 256
    number_of_workers :int = 8
    validation_size : int = 5000

    teacher_epochs:int = 50
    student_epochs:int = 50
    kd_epochs:int = 50

    learning_rate:float = 1e-3
    minimum_learning_rate:float = 1e-5
    weight_decay:float = 5e-4

    temperature:float = 4.0
    hard_loss_weight:float = 0.5

    split_seed: int =42
    teacher_seed :int =100
    student_seed : int =200

    
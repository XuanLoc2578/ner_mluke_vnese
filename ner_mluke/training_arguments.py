"""Arguments for training."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    read_file_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "raw file directory"
        }
    )
    write_file_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "written raw file directory"
        }
    )
    train_dataset_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "File path to train documents"
        }
    )
    val_dataset_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "File path to validation documents"
        }
    )
    test_dataset_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "File path to test documents"
        }
    )


    num_labels: Optional[str] = field(
        default=5,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name_or_path"
        },
    )
    # config_file: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "Path to config file"},
    # )
    # tokenizer_name: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "Pretrained tokenizer name or path if not the same as model_name_or_path"
    #     },
    # )



@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    max_seq_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total context's sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    lr: Optional[float] = field(
        default=0.00005,
        metadata={
            "help": "Learning rate"
        },
    )
    epochs: Optional[int] = field(
        default=3,
        metadata={
            "help": "How many time to travel through all samples in the dataset"
        },
    )
    batch_size: Optional[int] = field(
        default=4,
        metadata={
            "help": "Nummber of batch_size"
        }
    )
    num_workers: Optional[int] = field(
        default=2,
        metadata={
            "help": "Number of workers"
        }
    )


## TODO split model args and data args:
## -model args: name, cache_dir, max_seq_len, lr, epochs, batch_size, num_labels, num_workers
## -dataa args: path_to_3_type_folders **/files (train, dev, test)
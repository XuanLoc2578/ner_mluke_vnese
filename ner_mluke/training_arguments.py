"""Arguments for training."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch."""
    pretrained_model_name_or_path: Optional[str] = field(
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

    max_seq_length: Optional[int] = field(
        default=512,
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
            "help": "Number of traverses through all samples in the dataset"
        },
    )

    batch_size: Optional[int] = field(
        default=4,
        metadata={
            "help": "Number of batch_size"
        }
    )

    num_workers: Optional[int] = field(
        default=2,
        metadata={
            "help": "Number of workers"
        }
    )

    num_labels: Optional[str] = field(
        default=5,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name_or_path"
        },
    )

    # custom_label2id: Optional[dict] = field(
    #     default=dict(
    #         O= 0,
    #         B-PER= 1,
    #         I-PER= 2,
    #         B-MISC= 3,
    #         I-MISC= 4,
    #         B-LOC= 5,
    #         B-ORG= 6,
    #         I-LOC= 8,
    #         I-ORG= 7
    #     ),
    #     metadata={
    #         "help": "List of label and corresponding id"
    #     }
    # )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    read_file_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Raw file directory"
        }
    )

    write_file_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Written raw file directory"
        }
    )

    train_dataset_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "File path to train documents"
        }
    )

    dev_dataset_file: Optional[str] = field(
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

    save_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to saved model and loss checkpoint"
        }
    )



## TODO split model args and data args: __done
## -model args: name, cache_dir, max_seq_len, lr, epochs, batch_size, num_labels, num_workers
## -dataa args: path_to_3_type_folders **/files (train, dev, test)
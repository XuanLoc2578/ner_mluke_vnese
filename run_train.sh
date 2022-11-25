#!/bin/bash

python ner_mluke/run_train.py \
    --pretrained_model_name_or_path studio-ousia/luke-large-finetuned-conll-2003 \
    --cache_dir mounts/model/mluke \
    --save_dir mounts/checkpoint/ \
    --train_dataset_file mounts/dataset/written_train_git.txt \
    --dev_dataset_file mounts/dataset/written_dev_git.txt \
    --test_dataset_file mounts/dataset/written_test_git.txt \
    --max_seq_length=256 \
    --epochs=3 \
    --batch_size=2 \
    --num_workers=2 \
    --lr=0.00005
#    --output_dir \
#    --read_file_dir mounts/dataset/dev_git.csv \
#    --write_file_dir mounts/dataset/written_dev_git.txt \

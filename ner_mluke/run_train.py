import matplotlib.pyplot as plt
import random
# import tqdm

import torch
import torch.optim as optim
import logging
# from torch import nn
# from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers.hf_argparser import HfArgumentParser

from training_arguments import ModelArguments, DataArguments
from mydataset import DataProcessor
from mymodel import ModelProcessor

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments)
    )
    model_args, data_args = parser.parse_args_into_dataclasses()

    lr = model_args.lr
    epochs = model_args.epochs

    model_processor = ModelProcessor()
    model, tokenizer = model_processor.model_and_tokenizer(model_args=model_args)

    data_processor = DataProcessor()
    train_dataloader = data_processor.dataloader(tokenizer=tokenizer,
                                                 model_args=model_args,
                                                 dataset_file=data_args.dev_dataset_file
                                                 )

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True)

    random.seed(42)

    train_loss_list = []
    model.train()

    for epoch in range(epochs):
        for step, train_batch in enumerate(train_dataloader):
            train_input_ids, train_attention_mask, train_entity_ids, train_entity_position_ids, train_entity_attention_mask, train_label_id = train_batch

            train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
            train_attention_mask = torch.tensor(train_attention_mask, dtype=torch.long)
            train_entity_ids = torch.tensor(train_entity_ids, dtype=torch.long)
            train_entity_position_ids = torch.tensor(train_entity_position_ids, dtype=torch.long)
            train_entity_attention_mask = torch.tensor(train_entity_attention_mask, dtype=torch.long)
            train_label_id = torch.tensor(train_label_id, dtype=torch.long)

            optimizer.zero_grad()

            outputs = model(input_ids=train_input_ids,
                            attention_mask=train_attention_mask,
                            entity_ids=train_entity_ids,
                            entity_position_ids=train_entity_position_ids,
                            entity_attention_mask=train_entity_attention_mask,
                            labels=train_label_id
                            )

            # logits = outputs.logits
            train_loss = outputs.loss
            train_loss_list.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
            scheduler.step(train_loss)

            if step % 50 == 49:
                print('training loss: ', train_loss_list[-1], '   step: ', step, '   epoch: ', epoch + 1)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, data_args.save_dir)

    for i in train_loss_list:
        print(i)
    plt.plot(train_loss_list)
    plt.show()


if __name__ == '__main__':
    main()

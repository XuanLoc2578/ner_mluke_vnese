import tqdm

import torch
import torch.optim as optim
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import transformers
from transformers import LukeForTokenClassification
from transformers.hf_argparser import HfArgumentParser

from training_arguments import ModelArguments, DataTrainingArguments
from mydataset import ReadAndWriteFile, DataProcessor



def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments)
    )
    model_args, data_args = parser.parse_args_into_dataclasses()

    dev_dataset_file = model_args.dev_dataset_file
    epochs = data_args.epochs
    lr = data_args.lr


    data_processor = DataProcessor()
    model_config, train_dataloader = data_processor.gen_function(dataset_file=dev_dataset_file)

    model = LukeForTokenClassification(model_config)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True)


    train_loss_list = []
    model.train()

    for epoch in range(epochs):
        for step, train_batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
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

        # model.eval()
        # with torch.set_grad_enabled(False):
        #     val_loss_list = []
        #     for step, val_batch in enumerate(tqdm(val_dataloader, desc="Iteration")):
        #         val_input_ids, val_attention_mask, val_entity_ids, val_entity_position_ids, val_entity_attention_mask, val_label_id = val_batch  # label_mask,
        #
        #         val_input_ids = torch.tensor(val_input_ids, dtype=torch.long)
        #         val_attention_mask = torch.tensor(val_attention_mask, dtype=torch.long)
        #         val_entity_ids = torch.tensor(val_entity_ids, dtype=torch.long)
        #         val_entity_position_ids = torch.tensor(val_entity_position_ids, dtype=torch.long)
        #         val_entity_attention_mask = torch.tensor(val_entity_attention_mask, dtype=torch.long)
        #         val_label_id = torch.tensor(val_label_id, dtype=torch.long)
        #
        #         outputs = model(input_ids=val_input_ids,
        #                         attention_mask=val_attention_mask,
        #                         entity_ids=val_entity_ids,
        #                         entity_position_ids=val_entity_position_ids,
        #                         entity_attention_mask=val_entity_attention_mask,
        #                         labels=val_label_id
        #                         )
        #
        #         val_loss = outputs.loss
        #         val_loss_list.append(val_loss.item())
        #
        #         if step % 10 == 9:
        #             print('validation loss: ', val_loss_list[-1], '   step: ', step, '   epoch: ', epoch + 1)


if __name__ == '__main__':
    main()
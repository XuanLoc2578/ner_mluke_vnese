import torch
import torch.optim as optim
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

import json

from mydataset import DataProcessor
from mymodel import ModelProcessor

logger = logging.getLogger(__name__)


def main():
    # config_dir = '/home/nxloc/PycharmProjects_clone/ner_mluke_vnese (copy)/config.json'
    config_dir = '/config.json'
    with open(config_dir, 'r') as openfile:
        json_object = json.load(openfile)

    lr = json_object["lr"]
    epochs = json_object["epochs"]
    train_dataset_file = json_object["train_dataset_file"]
    dev_dataset_file = json_object["dev_dataset_file"]
    save_dir = json_object["save_dir"]
    print("dev_dataset_file: {}".format(dev_dataset_file))

    torch.manual_seed(42)

    model_processor = ModelProcessor(config_dir=config_dir)
    model, tokenizer = model_processor.model_and_tokenizer()

    data_processor = DataProcessor(config_dir=config_dir)
    train_dataloader = data_processor.dataloader(tokenizer=tokenizer,
                                                 dataset_file=dev_dataset_file
                                                 )
    # val_dataloader = data_processor.dataloader(tokenizer=tokenizer,
    #                                            dataset_file=dev_dataset_file
    #                                            )

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True)

    train_loss_list, val_loss_list = [], []

    for epoch in range(epochs):
        print('epoch {}'.format(epoch))
        model.train()
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

            train_loss = outputs.loss
            train_loss_list.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
            scheduler.step(train_loss)

            if step % 1000 == 999:
                print(" * train loss: {}, step: {}, epoch: {}".format(train_loss_list[-1], step, epoch + 1))

        path = '{}/checkpoint_{}.pt'.format(save_dir, epoch)

        torch.save({
            'model_state_dict': model.state_dict(),
        }, path)

    for i in train_loss_list:
        print(i)

        # model.eval()
        # with torch.set_grad_enabled(False):
        #     for step, val_batch in enumerate(val_dataloader):
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
        #         if step % 500 == 499:
        #             print("validation loss: {}, step: {}, epoch: {}".format(val_loss_list[-1], step, epoch + 1))


if __name__ == '__main__':
    main()

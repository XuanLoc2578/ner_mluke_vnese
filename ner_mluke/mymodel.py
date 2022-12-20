import json

import torch
from transformers import LukeForTokenClassification, MLukeTokenizer, AutoTokenizer



class ModelProcessor:
    def __init__(self, config_dir):
        with open(config_dir, 'r') as openfile:
            json_object = json.load(openfile)

        # self.model_config_dir = json_object["model_config_dir"]
        self.model_dir = json_object["model_dir"]
        self.model_tokenizer_dir = json_object["model_tokenizer_dir"]
        self.num_labels = json_object["num_labels"]
        self.save_dir = json_object["save_dir"]
        self.max_seq_length = json_object["max_seq_length"]

    def model_and_tokenizer(self):
        model, tokenizer = self.load_pretrained_model(self.model_dir, self.model_tokenizer_dir)

        return model, tokenizer

    def load_pretrained_model(self, model_dir, model_tokenizer_dir):
        print("Creating model and tokenizer")
        model = LukeForTokenClassification.from_pretrained(model_dir,
                                                                 local_files_only=True,
                                                                 ignore_mismatched_sizes=True
                                                                 )
        tokenizer = MLukeTokenizer.from_pretrained(model_tokenizer_dir,
                                                   local_files_only=True,
                                                   ignore_mismatched_sizes=True
                                                   )

        return model, tokenizer


class Inference(ModelProcessor):
    def __init__(self, config_dir, checkpoint_ind):
        super().__init__(config_dir=config_dir)
        self.model, self.tokenizer = self.model_and_tokenizer()

        PATH = self.save_dir + "checkpoint_" + checkpoint_ind + ".pt"
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    # def preprocess(self, text):
    #     start_pos, end_pos = [], []
    #     word_list = text.split(' ')
    #     for i in range(len(word_list)):
    #         ind = text.find(word_list[i])
    #         start_pos.append(ind)
    #         end_pos.append(ind + len(word_list[i]) - 1)
    #
    #     entity_spans = []
    #     # for j, s_pos in enumerate(start_pos):
    #     #     for e_pos in end_pos[j:]:
    #     #         entity_spans.append((s_pos, e_pos))
    #     for idx in range(len(start_pos)):
    #         entity_spans.append((start_pos[idx], end_pos[idx]))
    #
    #     return entity_spans

    def infer(self, text):
        # entity_spans = self.preprocess(text)
        inputs = self.tokenizer(text,
                                return_tensors="pt",
                                truncation=True,
                                padding="max_length",
                                return_attention_mask=True,
                                max_length=self.max_seq_length,
                                add_special_tokens=True
                                )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # entity_ids = inputs["entity_ids"]
        # entity_position_ids = inputs["entity_position_ids"]
        # entity_attention_mask = inputs["entity_attention_mask"]
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             # entity_ids=entity_ids,
                             # entity_position_ids=entity_position_ids,
                             # entity_attention_mask=entity_attention_mask,
                             )
        logits = outputs.logits
        print(logits.shape)
        predicted_class_indices = logits.argmax(-1).squeeze().tolist()

        for i in range(1, len(predicted_class_indices) - 1):
            if predicted_class_indices[i] != 9:
                tmp_list = [input_ids[0][i]]
                for j in range(i + 1, len(predicted_class_indices) - 1):
                    if predicted_class_indices[j] == 9:
                        tmp_list.append(input_ids[0][j])
                    else:
                        word_tensor = torch.tensor(tmp_list)
                        break

                print("{}: {}".format(self.tokenizer.decode(word_tensor), self.model.config.id2label[predicted_class_indices[i]]))

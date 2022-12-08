import json

import torch
from transformers import LukeForTokenClassification, LukeTokenizer, AutoConfig


class ModelProcessor:
    def __init__(self, config_dir):
        with open(config_dir, 'r') as openfile:
            json_object = json.load(openfile)

        self.model_config_dir = json_object["model_config_dir"]
        self.model_tokenizer_dir = json_object["model_tokenizer_dir"]
        self.num_labels = json_object["num_labels"]
        self.save_dir = json_object["save_dir"]

    def model_and_tokenizer(self):
        custom_config = self.custom_model_config(self.model_config_dir)
        model, tokenizer = self.load_pretrained_model(self.model_tokenizer_dir, custom_config)

        return model, tokenizer

    def custom_model_config(self, model_config_dir):
        print("Creating custom config")
        custom_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_config_dir)
        custom_config.num_labels = self.num_labels
        custom_config.label2id = {'O': 0,
                                  'B-PER': 1,
                                  'I-PER': 2,
                                  'B-MISC': 3,
                                  'I-MISC': 4,
                                  'B-LOC': 5,
                                  'B-ORG': 6,
                                  'I-ORG': 7,
                                  'I-LOC': 8
                                  }
        custom_config.id2label = {0: 'O',
                                  1: 'B-PER',
                                  2: 'I-PER',
                                  3: 'B-MISC',
                                  4: 'I-MISC',
                                  5: 'B-LOC',
                                  6: 'B-ORG',
                                  7: 'I-ORG',
                                  8: 'I-LOC'
                                  }

        return custom_config

    def load_pretrained_model(self, model_tokenizer_dir, custom_config):
        print("Creating model and tokenizer")
        model = LukeForTokenClassification(custom_config)
        tokenizer = LukeTokenizer.from_pretrained(model_tokenizer_dir,
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

    def preprocess(self, text):
        start_pos, end_pos = [], []
        word_list = text.split(' ')
        for i in range(len(word_list)):
            ind = text.find(word_list[i])
            start_pos.append(ind)
            end_pos.append(ind + len(word_list[i]) - 1)

        entity_spans = []
        for j, s_pos in enumerate(start_pos):
            for e_pos in end_pos[j:]:
                entity_spans.append((s_pos, e_pos))

        return entity_spans

    def infer(self, text):
        entity_spans = self.preprocess(text)
        inputs = self.tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        entity_ids = inputs["entity_ids"]
        entity_position_ids = inputs["entity_position_ids"]
        entity_attention_mask = inputs["entity_attention_mask"]
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             entity_ids=entity_ids,
                             entity_position_ids=entity_position_ids,
                             entity_attention_mask=entity_attention_mask,
                             )
        logits = outputs.logits
        predicted_class_indices = logits.argmax(-1).squeeze().tolist()
        for span, predicted_class_idx in zip(entity_spans, predicted_class_indices):
            if predicted_class_idx != 0:
                print(text[span[0]: span[1]], self.model.config.id2label[predicted_class_idx])





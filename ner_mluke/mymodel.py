import json
from transformers import LukeForTokenClassification, LukeTokenizer, AutoConfig


class ModelProcessor:
    def __init__(self, config_dir):
        with open(config_dir, 'r') as openfile:
            json_object = json.load(openfile)

        self.pretrained_model_name_or_path = json_object["pretrained_model_name_or_path"]
        self.cache_dir = json_object["cache_dir"]
        self.num_labels = json_object["num_labels"]

    def model_and_tokenizer(self):
        custom_config = self.custom_model_config(self.pretrained_model_name_or_path)
        model, tokenizer = self.load_pretrained_model(self.pretrained_model_name_or_path, custom_config, self.cache_dir)

        return model, tokenizer

    def custom_model_config(self, pretrained_model_name_or_path):
        custom_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
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

    def load_pretrained_model(self, pretrained_model_name_or_path, custom_config, cache_dir):
        model = LukeForTokenClassification(custom_config)
        tokenizer = LukeTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=cache_dir)

        return model, tokenizer


# class Inference:
#     def __init__(self, model_args, custom_config):
#         self.model_args = model_args
#         model_processor = ModelProcessor()
#         self.model, self.tokenizer = model_processor.model_and_tokenizer(model_args=self.model_args)
#
#     def prepare_inputs(self, text):
#         start_position, end_position = [], []
#
#         word_list = [i for i in text.split(' ')]
#         for i in range(len(word_list)):
#             ind = text.find(word_list[i])
#             start_position.append(ind)
#             end_position.append(ind + len(word_list[i]) - 1)
#
#         entity_spans = []
#         for i, start_pos in enumerate(start_position):
#             for end_pos in end_position[i:]:
#                 entity_spans.append((start_pos, end_pos))
#
#         return entity_spans
#
#     def infer(self, text):
#         entity_spans = self.prepare_inputs(text=text)
#         inputs = self.tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
#         outputs = self.model(inputs['input_ids'],
#                              inputs['attention_mask'],
#                              inputs['entity_ids'],
#                              inputs['entity_position_ids'],
#                              inputs['entity_attention_mask']
#                              )
#         logits = outputs.logits
#         predicted_class_indices = logits.argmax(-1).squeeze().tolist()
#         for span, predicted_class_idx in zip(entity_spans, predicted_class_indices):
#             if predicted_class_idx != 0:
#                 print(text[span[0]: span[1]], self.model.config.id2label[predicted_class_idx])

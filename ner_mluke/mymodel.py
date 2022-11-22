import transformers
from transformers import LukeForTokenClassification, LukeTokenizer, AutoConfig
import training_arguments


class ModelProcessor:
    def model_and_tokenizer(self, model_args):
        pretrained_model_name_or_path = model_args.pretrained_model_name_or_path
        custom_config = self.custom_model_config(pretrained_model_name_or_path)
        model, tokenizer = self.load_pretrained_model(pretrained_model_name_or_path, custom_config)

        # print('*'*80)
        # print('pretrained_model_name_or_path: {}'.format(pretrained_model_name_or_path))

        return model, tokenizer

    def custom_model_config(self, pretrained_model_name_or_path):
        custom_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        custom_config.num_labels = 9
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

        return custom_config

    def load_pretrained_model(self, pretrained_model_name_or_path, custom_config):
        model = LukeForTokenClassification(custom_config)
        tokenizer = LukeTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)

        return model, tokenizer


class Inference:
    def __init__(self, model_name_or_path):
        self.tokenizer = LukeTokenizer.from_pretrained(model_name_or_path)
        self.model = LukeForTokenClassification.from_pretrained(model_name_or_path)

    def infer(self, text):
        start_position, end_position = [], []

        word_list = [i for i in text.split(' ')]
        for i in range(len(word_list)):
            ind = text.find(word_list[i])
            start_position.append(ind)
            end_position.append(ind + len(word_list[i]) - 1)

        entity_spans = []
        for i, start_pos in enumerate(start_position):
            for end_pos in end_position[i:]:
                entity_spans.append((start_pos, end_pos))

        inputs = self.tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
        outputs = self.model(inputs['input_ids'],
                             inputs['attention_mask'],
                             inputs['entity_ids'],
                             inputs['entity_position_ids'],
                             inputs['entity_attention_mask']
                             )
        logits = outputs.logits
        predicted_class_indices = logits.argmax(-1).squeeze().tolist()
        for span, predicted_class_idx in zip(entity_spans, predicted_class_indices):
            if predicted_class_idx != 0:
                print(text[span[0]: span[1]], self.model.config.id2label[predicted_class_idx])



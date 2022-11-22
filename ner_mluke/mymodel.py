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



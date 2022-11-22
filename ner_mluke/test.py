import transformers
from transformers import LukeTokenizer, LukeForTokenClassification

# tknz = LukeTokenizer.from_pretrained(pretrained_model_name_or_path='studio-ousia/luke-large-finetuned-conll-2003')
model = LukeForTokenClassification.from_pretrained(pretrained_model_name_or_path='studio-ousia/luke-large-finetuned-conll-2003')

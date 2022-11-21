import tqdm
import unicodedata

import torch
from torch.utils.data import Dataset as Dataset, DataLoader as DataLoader, TensorDataset
import transformers
from transformers import AutoConfig, LukeForEntitySpanClassification, LukeTokenizer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

import training_arguments, feature_compress
from feature_compress import FeatureExtracted


class ReadAndWriteFile:
    def read_file(self, read_file_dir):
        read_file_dir = training_arguments.ModelArguments.read_file_dir
        with open(read_file_dir, 'r+') as f:
            lines = f.readlines()

        text = '-DOCSTART-\t-X-\t-X-\tO\n\n'
        count = 0
        for i in range(3, len(lines) - 1):
            if lines[i] == '\n':
                count += 1
            if count == 2:
                lines.insert(i + 1, text)
                count = 0
        lines.append('\n\n' + text)

        return lines

    def write_file(self, lines, write_file_dir):
        write_file_dir = training_arguments.ModelArguments.write_file_dir
        lines = self.read_file(training_arguments.ModelArguments.read_file_dir)

        if write_file_dir is not None:
            with open(write_file_dir, 'w') as f:
                for line in lines:
                    f.write(f"{line}")


class DataProcessor:

    def gen_function(self, dataset_file):
        model_name_or_path = training_arguments.ModelArguments.model_name_or_path
        max_seq_length = training_arguments.DataTrainingArguments.max_seq_length
        batch_size = training_arguments.DataTrainingArguments.batch_size
        num_workers = training_arguments.DataTrainingArguments.num_workers


        tokenizer = self.huggingface_download(model_name_or_path=model_name_or_path)
        documents = self.load_documents(dataset_file=dataset_file)
        examples = self.load_examples(max_seq_length=max_seq_length, documents=documents, tokenizer=tokenizer)
        model_config, custom_label2id = self.custom_model_config(model_name_or_path=model_name_or_path, documents=documents)
        final_tag_list = self.create_tag_list(documents=documents, examples=examples)

        label_id_list = self.create_label_id(
            examples=examples,
            tag_list=final_tag_list,
            custom_label2id=custom_label2id,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length
        )
        params_list = self.create_list_params(examples=examples, tokenizer=tokenizer, max_seq_length=max_seq_length)
        dataloader = self.create_dataset_and_dataloader(params_list=params_list,
                                                        label_id_tensor=label_id_list,
                                                        batch_size=batch_size,
                                                        num_workers=num_workers
                                                        )
        return model_config, dataloader


    def huggingface_download(self, model_name_or_path):
        cache_dir = training_arguments.ModelArguments.cache_dir
        print('model_name_or_path: {} \n cache_dir: {}'.format(model_name_or_path, cache_dir))
        tokenizer = LukeTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)

        return tokenizer

    def load_documents(self, dataset_file) -> list[dict]:
        line = '-DOCSTART-  -X- -X-	O'
        documents = []
        words = []
        labels = []
        sentence_boundaries = []
        with open(dataset_file) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("-DOCSTART-"):
                    if words:
                        documents.append(dict(
                            words=words,
                            labels=labels,
                            sentence_boundaries=sentence_boundaries
                        ))
                        words = []
                        labels = []
                        sentence_boundaries = []
                    continue

                if not line:  # chỗ này cho '\n\n'
                    if not sentence_boundaries or len(words) != sentence_boundaries[-1]:
                        sentence_boundaries.append(len(words))
                else:  # các line bình thường
                    items = line.split("\t")
                    words.append(items[0])
                    labels.append(items[-1])

        if words:
            documents.append(dict(
                words=words,
                labels=labels,
                sentence_boundaries=sentence_boundaries
            ))

        return documents

    def load_examples(self, max_seq_length, documents, tokenizer) -> list[dict]:
        examples = []
        # max_token_length = 510
        max_mention_length = 30

        for ind, document in enumerate(tqdm(documents)):
            words = document["words"]
            subword_lengths = [len(tokenizer.tokenize(w)) for w in words]
            total_subword_length = sum(subword_lengths)
            sentence_boundaries = document["sentence_boundaries"]

            for i in range(len(sentence_boundaries) - 1):
                sentence_start, sentence_end = sentence_boundaries[i:i + 2]
                if total_subword_length <= max_seq_length:
                    # if the total sequence length of the document is shorter than the
                    # maximum token length, we simply use all words to build the sequence
                    context_start = 0
                    context_end = len(words)

                else:
                    # if the total sequence length is longer than the maximum length, we add
                    # the surrounding words of the target sentence　to the sequence until it
                    # reaches the maximum length
                    context_start = sentence_start
                    context_end = sentence_end
                    cur_length = sum(subword_lengths[context_start:context_end])

                    while True:
                        if context_start > 0:
                            if cur_length + subword_lengths[context_start - 1] <= max_seq_length:
                                cur_length += subword_lengths[context_start - 1]
                                context_start -= 1
                            else:
                                break
                        if context_end < len(words):
                            if cur_length + subword_lengths[context_end] <= max_seq_length:
                                cur_length += subword_lengths[context_end]
                                context_end += 1
                            else:
                                break

                text = ""
                for word in words[context_start:sentence_start]:
                    if word[0] == "'" or (len(word) == 1 and self.is_punctuation(word)):
                        text = text.rstrip()
                    text += word
                    text += " "

                sentence_words = words[sentence_start:sentence_end]
                sentence_subword_lengths = subword_lengths[sentence_start:sentence_end]

                word_start_char_positions = []
                word_end_char_positions = []
                for word in sentence_words:
                    if word[0] == "'" or (len(word) == 1 and self.is_punctuation(word)):
                        text = text.rstrip()
                    word_start_char_positions.append(len(text))
                    text += word
                    word_end_char_positions.append(len(text))
                    text += " "

                for word in words[sentence_end:context_end]:
                    if word[0] == "'" or (len(word) == 1 and self.is_punctuation(word)):
                        text = text.rstrip()
                    text += word
                    text += " "
                text = text.rstrip()

                entity_spans = []
                original_word_spans = []
                for word_start in range(len(sentence_words)):
                    for word_end in range(word_start, len(sentence_words)):
                        if sum(sentence_subword_lengths[word_start:word_end]) <= max_mention_length:
                            entity_spans.append(
                                (word_start_char_positions[word_start], word_end_char_positions[word_end])
                            )
                            original_word_spans.append(
                                (word_start, word_end + 1)
                            )

                examples.append(dict(
                    text=text,
                    words=sentence_words,
                    entity_spans=entity_spans,
                    original_word_spans=original_word_spans,
                ))

        return examples

    def is_punctuation(self, char):
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

#####

    def custom_model_config(self, model_name_or_path, documents):
        model_config = AutoConfig.from_pretrained(model_name_or_path)

        unique_tag = []
        for i in range(len(documents)):
            for j in range(len(documents[i]['labels'])):
                if documents[i]['labels'][j] not in unique_tag:
                    unique_tag.append(documents[i]['labels'][j])

        custom_id2label = {i: label for i, label in enumerate(unique_tag)}
        custom_label2id = {label: i for i, label in enumerate(unique_tag)}

        model_config.id2label = custom_id2label
        model_config.label2id = custom_label2id

        return model_config, custom_label2id

    def create_tag_list(self, documents, examples) -> list[list]:
        final_word_list = []
        for i in range(len(examples)):
            final_word_list.append(examples[i]['words'])

        list_all_tag = []
        for i in range(len(documents)):
            for j in range(len(documents[i]['labels'])):
                list_all_tag.append(documents[i]['labels'][j])

        final_tag_list = []
        ind = 0
        for i in range(len(final_word_list)):
            tmp_list = list_all_tag[ind:ind + len(final_word_list[i])]
            final_tag_list.append(tmp_list)
            ind += len(final_word_list[i])

        return final_tag_list

    def create_list_params(self, examples, max_seq_length, tokenizer) -> object:
        list_input_ids, list_attention_mask, list_entity_ids, list_entity_position_ids, list_entity_attention_mask = [], [], [], [], []

        for i in range(len(examples)):
            source_encoding = tokenizer(
                text=' '.join(examples[i]["Word"]),
                entity_spans=examples[i]["Span"],
                max_length=max_seq_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                # return_tensors="pt"
            )

            for j in range(len(source_encoding['entity_position_ids'])):
                if len(source_encoding['entity_position_ids'][j]) > 30:
                    source_encoding['entity_position_ids'][j] = source_encoding['entity_position_ids'][j][:30]

            list_input_ids.append(source_encoding["input_ids"])
            list_attention_mask.append(source_encoding["attention_mask"])
            list_entity_ids.append(source_encoding["entity_ids"])
            list_entity_position_ids.append(source_encoding["entity_position_ids"])
            list_entity_attention_mask.append(source_encoding["entity_attention_mask"])

        list_input_ids_tensor = torch.tensor([f for f in list_input_ids])
        list_attention_mask_tensor = torch.tensor([f for f in list_attention_mask])
        list_entity_ids_tensor = torch.tensor([f for f in list_entity_ids])
        list_entity_position_ids_tensor = torch.tensor([f for f in list_entity_position_ids])
        list_entity_attention_mask_tensor = torch.tensor([f for f in list_entity_attention_mask])

        list_params = FeatureExtracted(
            list_input_ids_tensor,
            list_attention_mask_tensor,
            list_entity_ids_tensor,
            list_entity_position_ids_tensor,
            list_entity_attention_mask_tensor
        )
        return list_params

    def create_label_id(self, examples, tag_list, custom_label2id, max_seq_length, tokenizer) -> list[list]:
        label_id = []
        final_word_list = []
        final_tag_list = []
        for i in range(len(examples)):
            final_word_list.append(examples[i]['words'])
            final_tag_list.append(examples[i]['entity_spans'])
        custom_label2id = custom_label2id
        label_id_tensor_list = []

        for i in range(len(final_word_list[:len(final_word_list)])):
            tmp_label_id = []

            for j in range(len(final_word_list[i])):
                token = tokenizer.tokenize(final_word_list[i][j])
                for t in range(len(token)):
                    if t == 0:
                        tmp_label_id.append(custom_label2id[final_tag_list[i][j]])
                    else:
                        tmp_label_id.append(0)

            if len(tmp_label_id) > max_seq_length - 2:
                tmp_label_id = tmp_label_id[:max_seq_length - 2]

            tmp_label_id.insert(1, 0)
            tmp_label_id.append(1)

            while len(tmp_label_id) < max_seq_length:
                tmp_label_id.append(0)

            label_id.append(tmp_label_id)

        label_id_tensor = torch.tensor([f for f in label_id])
        return label_id_tensor

    def create_dataset_and_dataloader(self, params_list, label_id_tensor, batch_size, num_workers):
        input_ids_tensor, attention_mask_tensor, entity_ids_tensor, entity_position_ids_tensor, entity_attention_mask_tensor = params_list

        dataset = TensorDataset(input_ids_tensor,
                                attention_mask_tensor,
                                entity_ids_tensor,
                                entity_position_ids_tensor,
                                entity_attention_mask_tensor,
                                label_id_tensor
                                )
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        return dataloader

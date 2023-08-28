import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DuEEDataset(Dataset):
    """DuEventExtraction"""

    def __init__(self, data_path, tag_path, tokenizer, max_len=512,not_test=True):
        #加载id2entity
        self.label_vocab = {}
        examples = []
        tokenized_examples = []

        for line in open(tag_path, 'r', encoding='utf-8'):
            value, key = line.strip('\n').split('\t')
            self.label_vocab[key] = int(value)

        # 加载准备好的训练集
        with open(data_path, 'r', encoding='utf-8') as fp:
            if not_test:
                next(fp)
            for line in fp.readlines():
                if not_test:
                    #next(fp)
                    words, labels = line.strip('\n').split('\t')
                    words = words.split('\002')
                    labels = labels.split('\002')
                    examples.append([words, labels])
                else:
                    line=json.loads(line)
                    words= line["text"]
                    examples.append([words])
        self.label_num = len(self.label_vocab)

        with tqdm(enumerate(examples), total=len(examples), desc="tokenizing...") as pbar:
            for i, example in pbar:
                tokenized_example = tokenizer.encode_plus(
                    list(example[0]),
                    padding="max_length",
                    max_length=max_len,
                    is_split_into_words=True,
                    truncation=True
                )
                # 把 label 补齐
                if not_test:
                    labels = example[1]
                    pad_len = max_len - 2 - len(labels)
                    if pad_len >= 0:
                        labels += ["PAD"] * pad_len
                    else:
                        labels = labels[:pad_len]
                    labels = ["PAD"] + labels + ["PAD"]
                    labels = [self.label_vocab.get(label, -1) for label in labels]

                    tokenized_example["labels"] = labels
                tokenized_examples.append(tokenized_example)

        self.examples = examples
        self.tokenized_examples = tokenized_examples

    def __len__(self):
        return len(self.tokenized_examples)

    def __getitem__(self, index):
        return self.tokenized_examples[index]


'''class DuEEClsDataset(Dataset):
    """DuEventExtraction"""

    def __init__(self, args, data_path, tag_path, tokenizer):
        # 加载id2entity
        self.label_vocab = {}
        examples = []
        tokenized_examples = []

        for line in open(tag_path, 'r', encoding='utf-8'):
            value, key = line.strip('\n').split('\t')
            self.label_vocab[key] = int(value)

        df = pd.read_csv(data_path, delimiter="\t", quoting=3)
        examples = df.values.tolist()

        with tqdm(enumerate(examples), total=len(examples), desc="tokenizing...") as pbar:
            for i, example in pbar:
                tokenized_example = tokenizer.encode_plus(
                    example[1],
                    padding="max_length",
                    max_length=args.max_len,
                    truncation=True
                )
                # 把 label 补齐
                labels = self.label_vocab.get(example[0], -1)
                tokenized_example["labels"] = labels
                tokenized_examples.append(tokenized_example)
        self.examples = examples
        self.tokenized_examples = tokenized_examples

    def __len__(self):
        return len(self.tokenized_examples)

    def __getitem__(self, index):
        return self.tokenized_examples[index]'''


def collate_fn(batch):
    max_len = max([sum(x['attention_mask']) for x in batch])
    seq_lens = torch.tensor([sum(x["attention_mask"]) for x in batch])
    all_input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch])
    all_token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in batch])
    all_attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in batch])
    all_labels = torch.tensor([x["labels"][:max_len] for x in batch])

    return {
        "all_input_ids": all_input_ids,
        "all_token_type_ids": all_token_type_ids,
        "all_attention_mask": all_attention_mask,
        "all_labels": all_labels,
        "all_seq_lens": seq_lens
    }

'''def cls_collate_fn(batch):
    max_len = max([sum(x['attention_mask']) for x in batch])
    seq_lens = torch.tensor([sum(x["attention_mask"]) for x in batch])
    all_input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch])
    all_token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in batch])
    all_attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in batch])
    all_labels = torch.tensor([x["labels"] for x in batch])

    return {
        "all_input_ids": all_input_ids,
        "all_token_type_ids": all_token_type_ids,
        "all_attention_mask": all_attention_mask,
        "all_labels": all_labels,
        "all_seq_lens": seq_lens
    }'''
def test_collate_fn(batch):
    max_len = max([sum(x['attention_mask']) for x in batch])
    seq_lens = torch.tensor([sum(x["attention_mask"]) for x in batch])
    all_input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch])
    all_token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in batch])
    all_attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in batch])
    #all_labels = torch.tensor([x["labels"] for x in batch])

    return {
        "all_input_ids": all_input_ids,
        "all_token_type_ids": all_token_type_ids,
        "all_attention_mask": all_attention_mask,
        #"all_labels": all_labels,
        "all_seq_lens": seq_lens
    }
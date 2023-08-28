"""
sequence labeling
"""
import argparse
import ast
import json
import os
import random
import warnings
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DuEEDataset,collate_fn,test_collate_fn
from transformers import BertTokenizer
from tqdm import tqdm
from utils import load_dict, read_by_lines, write_by_lines
from model import Bert_BiLSTM_CRF
from metric import ChunkEvaluator
from torch.autograd import Variable
# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=8, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--tag_path", type=str, default="./conf/DuEE1.0/trigger_tag.dict", help="tag set path")
parser.add_argument("--train_data", type=str, default="./data/DuEE1.0/trigger/train.tsv", help="train data")
parser.add_argument("--dev_data", type=str, default="./data/DuEE1.0/trigger/dev.tsv", help="dev data")
parser.add_argument("--test_data", type=str, default="./data/DuEE1.0/trigger/test.tsv", help="test data")
#预测的数据
parser.add_argument("--predict_data", type=str, default="./data/DuEE1.0/duee_test2.json", help="predict data")
parser.add_argument("--do_train", type=ast.literal_eval, default=True, help="do train")
parser.add_argument("--do_predict", type=ast.literal_eval, default=True, help="do predict")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--valid_step", type=int, default=100, help="validation step")
parser.add_argument("--skip_step", type=int, default=20, help="skip step")
parser.add_argument("--batch_size", type=int, default=8, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoints", type=str, default="./ckpt/DuEE1.0/trigger", help="Directory to model checkpoint")
parser.add_argument("--init_ckpt", type=str, default="./ckpt/DuEE1.0/best.pdparams", help="already pretraining model checkpoint")
#预测结果地址
parser.add_argument("--predict_save_path", type=str, default="./ckpt/DuEE1.0/trigger/bert_test_pred.json", help="predict data save path")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")

#模型参数

parser.add_argument('--hidden_dim', type=int, default=500, help="hidden_dim")
parser.add_argument('--rnn_layers', type=int, default=1, help="rnn_layers")
parser.add_argument('--lstm_dropout_ratio', type=float, default=0.3, help="lstm_dropout_ratio")
parser.add_argument('--dropout', type=float, default=0.3, help="dropout")
args = parser.parse_args()
def evaluate(metric_collection, model, num_label, data_loader, criterion):
    model = model.cuda()
    model.eval()
    metric_collection.reset()
    batch_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            for key in batch.keys():
                batch[key] = batch[key].cuda()
            out,loss = model(input_ids=batch['all_input_ids'],
                            label_ids=batch['all_labels'],
                            attention_mask=batch['all_attention_mask'])
            batch_loss += loss.item()
            preds=out
            n_infer, n_label, n_correct = metric_collection.compute(batch["all_seq_lens"], preds, batch['all_labels'])
            metric_collection.update(n_infer, n_label, n_correct)

    precision, recall, f1_score = metric_collection.accumulate()
    model.train()
    return precision, recall, f1_score, batch_loss / (step + 1)


def set_seed(args):
    """sets random seed"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

def do_train():
    set_seed(args)
    torch.cuda.set_device(0)
    
    ignore_label = -1
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    label_map = load_dict(args.tag_path)
    
    model = Bert_BiLSTM_CRF(
        bert_config="bert-base-chinese", 
        tagset_size=len(label_map),
        embedding_dim=768,
        hidden_dim=500,
        rnn_layers=1,
        lstm_dropout_ratio=0.3,
        dropout=0.3
        )
    #bert_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, lstm_dropout_ratio, dropout

    print("============load data==========")
    train_ds = DuEEDataset(args.train_data, args.tag_path,tokenizer,args.max_seq_len)
    dev_ds = DuEEDataset(args.dev_data, args.tag_path,tokenizer,args.max_seq_len)
    
    #参数collate_fn是表示如何取样本的
    train_loader = DataLoader(dataset=train_ds,
                                        collate_fn=collate_fn,
                                        batch_size=args.batch_size,
                                        shuffle=True)
    dev_loader = DataLoader(dataset=dev_ds,
                                      batch_size=args.batch_size,
                                      collate_fn=collate_fn,
                                        shuffle=False)

    num_training_steps = len(train_loader) * args.num_epoch

    optimizer = torch.optim.Adam(#optimizer_grouped_parameters,
        params=model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay        )

    metric_collection = ChunkEvaluator(label_map, suffix=False)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)

    step, best_f1 = 0, 0.0
    model = model.cuda()
    model.train()
    for epoch in range(args.num_epoch):
        for step, batch in enumerate(train_loader):
            for key in batch.keys():
                batch[key] = batch[key].cuda()
            inputs=batch['all_input_ids']
            masks=batch['all_attention_mask']
            label_ids=batch['all_labels']
            inputs, masks, label_ids = Variable(inputs), Variable(masks), Variable(label_ids)
            out,loss = model(input_ids=inputs,
                        attention_mask=masks,
                        label_ids=label_ids)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_item = loss.item()
            if step > 0 and step % args.skip_step == 0:
                print(
                    f'train epoch: {epoch} - step: {step} (total: {num_training_steps}) - loss: {loss_item:.6f}'
                )
            #验证集合检验
            if step > 0 and step % args.valid_step == 0:
                p, r, f1, avg_loss = evaluate(metric_collection, model, len(label_map), dev_loader, criterion)
                print(f'dev step: {step} - loss: {avg_loss:.5f}, precision: {p:.5f}, recall: {r:.5f}, f1: {f1:.5f} current best {best_f1:.5f}')
                if f1 > best_f1:
                    best_f1 = f1
                    print(f'==============================================save best model best performerence {best_f1:5f}')
                    torch.save(model.state_dict(), f'{args.checkpoints}/best.pdparams')

def do_predict():
    torch.cuda.set_device(0)
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese",)
    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}

    model = Bert_BiLSTM_CRF(
        bert_config="bert-base-chinese", 
        tagset_size=len(label_map),
        embedding_dim=768,
        hidden_dim=500,
        rnn_layers=1,
        lstm_dropout_ratio=0.5,
        dropout=0.5
        )

    print("============start predict==========")
    if not args.init_ckpt or not os.path.isfile(args.init_ckpt):
        raise Exception("init checkpoints {} not exist".format(args.init_ckpt))
    else:
        state_dict = torch.load(args.init_ckpt)
        model.load_state_dict(state_dict)
        print("Loaded parameters from %s" % args.init_ckpt)

    #test文件
    sentences = read_by_lines(args.predict_data)  # origin data format
    sentences = [json.loads(sent) for sent in sentences]
    #test文件
    test_ds = DuEEDataset(args.predict_data, args.tag_path, tokenizer, args.max_seq_len,not_test=False)
    # 加载批数据
    test_loader = DataLoader(dataset=test_ds,
                              collate_fn=test_collate_fn,
                              batch_size=args.batch_size)

    results = []
    model=model.cuda()
    model.eval()
    #预测过程
    #for batch in batch_encoded_inputs:
    for batch in tqdm(test_loader, total=len(test_loader)):
        for key in batch.keys():
                batch[key] = batch[key].cuda()
        out = model(input_ids=batch['all_input_ids'], attention_mask=batch['all_attention_mask'])
        probs_ids=out
        for p_ids, seq_len in zip(probs_ids, batch["all_seq_lens"]):
            label_one = [id2label[pid] for pid in p_ids[1: seq_len - 1]]
            results.append({"labels": label_one})
    assert len(results) == len(sentences)

    for sent, ret in zip(sentences, results):
        sent["pred"] = ret
    sentences = [json.dumps(sent, ensure_ascii=False) for sent in sentences]
    write_by_lines(args.predict_save_path, sentences)
    print("save data {} to {}".format(len(sentences), args.predict_save_path))

if __name__ == '__main__':
    if args.do_train:
        do_train()
    elif args.do_predict:
        do_predict()

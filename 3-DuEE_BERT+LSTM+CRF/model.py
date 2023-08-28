from transformers import BertModel
from torchcrf import CRF
import torch
import torch.nn as nn
from torch.autograd import Variable

class Bert_BiLSTM_CRF(nn.Module):
    """
    bert_lstm_crf model
    embedding_dim
    hidden_dim
    rnn_layers
    dropout_ratio
    dropout1,
    """
    def __init__(self, bert_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, lstm_dropout_ratio, dropout):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.word_embeds = BertModel.from_pretrained(bert_config)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.tagset_size = tagset_size
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=rnn_layers, bidirectional=True, dropout=lstm_dropout_ratio, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.liner = nn.Linear(hidden_dim*2, tagset_size)
        self.crf = CRF(num_tags=tagset_size,batch_first=True)
        

    def rand_init_hidden(self, batch_size):
        
        '''return Variable(
            torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)), Variable(
            torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim))'''
        return torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim),torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)
    def neg_log_likelihood(self, logits=None, label_ids=None, attention_mask=None):# 损失函数
            return -self.crf(emissions=logits, tags=label_ids, mask=attention_mask)
    def forward(self, 
                input_ids=None,
                label_ids=None,
                attention_mask=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)
        attention_mask=attention_mask.bool()
        output = self.word_embeds(input_ids, attention_mask=attention_mask)
        embeds, pooled_output = output[0], output[1]
        h,c = self.rand_init_hidden(batch_size)
        h = h.cuda()
        c = c.cuda()
        lstm_out, hidden = self.lstm(embeds, (h,c))
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*2)#复制
        d_lstm_out = self.dropout(lstm_out)
        l_out = self.liner(d_lstm_out)
        lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)
        if label_ids!=None:
            '''print(lstm_feats.shape)
            print(label_ids.shape)
            print(attention_mask.shape)'''
            loss_value = self.neg_log_likelihood(lstm_feats, label_ids,attention_mask)
            batch_size = lstm_feats.size(0)
            loss_value /= float(batch_size) 
            out = self.crf.decode(emissions=lstm_feats,mask=attention_mask)
            return out,loss_value
        else:
            out = self.crf.decode(emissions=lstm_feats,mask=attention_mask)
        return out
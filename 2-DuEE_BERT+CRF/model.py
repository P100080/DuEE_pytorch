from torch import nn
import torch
from transformers import BertModel
from torchcrf import CRF
from torch.autograd import Variable
class Bert_CRF(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        super(Bert_CRF, self).__init__()
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.crf = CRF(num_tags=num_classes,batch_first=True)     

    def neg_log_likelihood(self, logits=None, label_ids=None, attention_mask=None):# 损失函数
            return -self.crf(emissions=logits, tags=label_ids, mask=attention_mask)
    
    def forward(self,
                input_ids=None,
                label_ids=None,
                token_type_ids=None,
                attention_mask=None):
        output = self.bert(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        
        sequence_output, pooled_output = output[0], output[1]
        logits = self.classifier(sequence_output)
        if label_ids!=None:
            attention_mask=attention_mask.bool()
            out = self.crf.decode(emissions=logits,mask=attention_mask)
            loss = self.neg_log_likelihood(logits=logits,label_ids=label_ids,attention_mask=attention_mask)
            return out,loss
        else:
            attention_mask=attention_mask.bool()
            out = self.crf.decode(emissions=logits,mask=attention_mask)
            return out

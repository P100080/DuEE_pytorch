from torch import nn
from transformers import BertModel

class DuEE_model(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        super(DuEE_model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        #self.dropout=nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None):
        output = self.bert(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        sequence_output, pooled_output = output[0], output[1]
        #out=self.dropout(sequence_output)
        logits = self.classifier(sequence_output)#sequence_output
        return logits

import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer
#from transformers import BertModel, BertTokenizer
import pdb

class Config(object):
    """
    Configuration
    """

    def __init__(self, dataset):
        self.model_name = 'bert'
        # Train
        self.train_path = dataset + '/data/train.txt'
        # Test
        self.test_path = dataset + '/data/test.txt'
        # Val
        self.dev_path = dataset + '/data/dev.txt'
        # Pickled dataset for fast load
        self.datasetpkl = dataset + '/data/dataset.pkl'
        # Classes
        self.class_list = [ x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        # Model checkpoints
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        # Devices
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Exit training if there is no improvement in 1000 batches
        self.require_improvement = 1000
        # Num of classes
        self.num_classes = len(self.class_list)
        # Num of epochs
        self.num_epochs = 1
        # Batch size
        self.batch_size = 128
        # length of every sentence (paragraph) , padding if less than it, cut it if longer than it. 
        self.pad_size = 32
        # Learning rate
        self.learning_rate = 1e-5
        # Location of pretrained Bert
        self.bert_path = 'bert_pretrain'
        # Bert tokenizer # Do not understand*********************************************************************
        #pdb.set_trace()
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        # Bert hidden layers 
        self.hidden_size = 768
        # Fine tune
        self.fine_tune = True


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        #self.bert = BertModel.from_pretrained('bert-base-chinese')
        for param in self.bert.parameters():
            param.requires_grad = config.fine_tune
        
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, token_ids, seq_len, mask):

        # input sentence shape: [batch_size, padd_size] = [128, 32]
        # Mask for paddings
        #shape of pool = [batch_size, hidden_size] = [128, 768]
        _, pooled = self.bert(input_ids=token_ids, attention_mask=mask)#, output_all_encoded_layers=False)
        # shape [batch_size, num_classes] = [128, 10]
        out = self.fc(pooled)

        return out


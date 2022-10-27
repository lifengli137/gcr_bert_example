from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import time
from datetime import timedelta
import pickle as pkl
import os
import requests
import tarfile
import pdb

PAD, CLS = '[PAD]', '[CLS]'

def load_dataset(data_path, config):
    """
    return:  4 lists: ids, label, ids_len, mask
    """
    line_num = 0
    contents = []
    with open(data_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):

            # For DDP
            # if line_num % config.world != 0:
            #     line_num += 1
            #     continue
            # line_num += 1

            line = line.strip()
            if not line:
                continue
            
            content, label = line.split('\t')
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            pad_size = config.pad_size
            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids)  + [0] * (pad_size - len(token))
                    token_ids = token_ids + ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            #[1, 12, 403, 291, 85, 18, 245, 74, 89, 147, 522, 218, 674, 208, 206, 26, 931, 654, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #'3'
            #19
            #[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            contents.append((token_ids, int(label), seq_len, mask))
    #pdb.set_trace()
    
    return contents
    
def tensorize_dataset(dataset):

    token_ids_tensor = torch.tensor([c[0] for c in dataset], dtype=torch.long)
    label_tensor = torch.tensor([c[1] for c in dataset], dtype=torch.long)
    seq_len_tensor = torch.tensor([c[2] for c in dataset], dtype=torch.long)
    mask_tensor = torch.tensor([c[3] for c in dataset], dtype=torch.long)
    tensor_dataset = TensorDataset(token_ids_tensor, label_tensor, seq_len_tensor, mask_tensor)

    return tensor_dataset

def build_dataset(config):
    """
    return: train, dev, test
    """
    if os.path.exists(config.datasetpkl):
        dataset = pkl.load(open(config.datasetpkl, 'rb'))
        train = dataset['train']
        dev = dataset['dev']
        test = dataset['test']
    else:
        train = load_dataset(config.train_path, config)
        dev = load_dataset(config.dev_path, config)
        test = load_dataset(config.test_path, config)

        train = tensorize_dataset(train)
        dev = tensorize_dataset(dev)
        test = tensorize_dataset(test)
        dataset = {}
        dataset['train'] = train
        dataset['dev'] = dev
        dataset['test'] = test
        pkl.dump(dataset, open(config.datasetpkl, 'wb'))

    return train, dev, test


def get_time_dif(start_time):
    """
    Get the time difference
    """

    end_time = time.time()
    time_dif = end_time - start_time

    return timedelta(seconds=int(round(time_dif)))





def build_dataloader(data, config):
    sampler = DistributedSampler(data, num_replicas=config.hvd.size(), rank=config.hvd.rank())
    dataloader = DataLoader(data, sampler=sampler, batch_size=config.batch_size)
    return dataloader



def download_model(config):
    
    
    url = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz'
    target_path = '/tmp/bert-base-chinese.tar.gz'
    print("Start to download model from {}".format(url))
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception("Unable to download model from {}.".format(url))
        
    with open(target_path, 'wb') as f_out:
        f_out.write(response.raw.read())

    print("Start to untar model to {}.".format(config.bert_path))
    with tarfile.open(target_path,"r:gz") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=config.bert_path)
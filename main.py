import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import horovod.torch as hvd

import numpy as np
from importlib import import_module
import argparse 
import utils
import train
import pdb

parser = argparse.ArgumentParser(description='GCR-Bert-Text-Classification.')
parser.add_argument('--model', type=str, default='bert', help='Choose a model: bert')
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--dataset", type=str, default="THUCNews")
parser.add_argument("--data_path", type=str, default=".")
parser.add_argument("--model_path", type=str, default=".")
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--warmup", type=float, default=0.05)



args = parser.parse_args()

if __name__ == '__main__':
    # path of dataset
    #pdb.set_trace()
    model_name = args.model
    x = import_module('models.'+ model_name)
    
    config = x.Config(args)
    

    config.hvd = hvd
    
    hvd.init()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    torch.cuda.set_device(hvd.local_rank())
    
    config.rank = hvd.rank()
    config.world = hvd.size()
    
    if hvd.local_rank() == 0:
        utils.download_model(config)
    hvd.broadcast_object(0, root_rank=0)
    model = x.Model(config)

    start_time = time.time()
    print('Loading dataset')
    train_data, dev_data, test_data = utils.build_dataset(config)
    
    train_iter = utils.build_dataloader(train_data, config)
    dev_iter = utils.build_dataloader(dev_data, config)
    test_iter = utils.build_dataloader(test_data, config)
    

    time_dif = utils.get_time_dif(start_time)
    print("Prepare data time: ", time_dif)

    # Train, eval, test
    model = model.to(config.device)
    
    if hvd.nccl_built() == False:
        raise Exception("NCCL was not compiled in Horovod!")

    
    train.train(config, model, train_iter, dev_iter, test_iter)

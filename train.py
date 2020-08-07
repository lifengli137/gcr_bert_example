
import numpy as np
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
from sklearn import metrics
import time
import pdb
import torch.optim as optim
from pytorch_pretrained.optimization import BertAdam

def train(config, model, train_iter, dev_iter, test_iter):
    """
    Train model
    """

    # get the model's parameters
    param_optimizer = list(model.named_parameters())
    # which parameters do not need to be decay
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
    ]
    optimizer = BertAdam(params=optimizer_grouped_parameters,
                        schedule = None,
                        lr = config.learning_rate,
                        warmup=0.05,
                        t_total=len(train_iter) * config.num_epochs)


    config.hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    config.hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    optimizer = config.hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    start_time = time.time()
    # activate BatchNormalization & dropout
    model.train()


   
    flag = False
    # progress
    total_batch = 0
    # Best loss in dev
    dev_best_loss = float('inf')
    # Last time loss was decreased's batch number in dev
    last_improve = 0
    # no improvement in long time, it's ok to stop train
    
    for epoch in range(config.num_epochs):
        if config.hvd.rank() == 0:
            print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (token_ids, label, seq_len, mask) in enumerate(train_iter):
            
            token_ids = token_ids.to(config.device)
            label_gpu = label.to(config.device)
            seq_len = seq_len.to(config.device)
            mask = mask.to(config.device)

            outputs = model(token_ids, seq_len, mask)
            model.zero_grad()
            loss = F.cross_entropy(outputs, label_gpu)
            loss.backward()
            optimizer.step()
            #Every n batches, go dev
            if total_batch % 100 == 0:
                # Get the highest one from Softmax()
                predit = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(label, predit)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    if config.hvd.rank() == 0:
                        torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = utils.get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.2}, Train Acc:{2:>6.2}, Val Loss: {3:>5.2}, Val Acc: {4:>6.2%}, Time: {5} {6}'
                if config.hvd.rank() == 0:
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch = total_batch + 1
            if total_batch - last_improve > config.require_improvement:
                # No improvement for too long (longer than config.require_improvement)
                print('No improvement for too long. Stop training automatically.')
                flag = True
                break

        if flag:
            break



def evaluate(config, model, dev_iter, test=False):
    """

    """

    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for token_ids, label, seq_len, mask in dev_iter:

            token_ids = token_ids.to(config.device)
            label_gpu = label.to(config.device)
            seq_len = seq_len.to(config.device)
            mask = mask.to(config.device)

            outputs = model(token_ids, seq_len, mask)
            loss = F.cross_entropy(outputs, label_gpu)
            loss_total = loss_total + loss
            label = label.numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all) 
        return acc, loss_total / len(dev_iter), report, confusion
           
    return acc, loss_total / len(dev_iter)
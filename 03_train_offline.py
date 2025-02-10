import numpy as np
import glob
import ecole
import argparse
import os
import warnings
import csv
import torch
import torch_geometric
import pickle5
import random
import time
from common.buffer import GraphDataset, TreeDataset
from common.utils import fix_seed, get_config
from evaluate import evaluate_ml
from brancher.models import create_model, save_model


def train(config):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = os.path.join(config.save_dir, config.task_name, 'trained_models')
    log_path = os.path.join(config.save_dir, config.task_name, 'logs')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    model, logger_keys, log_file = create_model(config, log_path, device)            
    
    if not config.debug:
        with open(log_file, 'w') as csvfile: 
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(logger_keys) 
           
    # training data  
    data_dir = os.path.join(config.dataset_dir, config.task_name, config.dataset_name) 
    train_files = [str(path) for path in glob.glob(data_dir + "/sample_*.pkl")]

    
    Dataset = TreeDataset if config.tree else GraphDataset

    prenorm_data = Dataset([file for i, file in enumerate(train_files) if i%10==0]) 
    prenorm_loader = torch_geometric.loader.DataLoader(prenorm_data, batch_size=config.batch_size*2, shuffle=True, num_workers=8)
    

    if config.warmup:
        print('Start prenorm')
        model.prenorm(prenorm_loader)
        print('Finish prenorm')
    
    for epoch in range(1, 1+config.epochs): 
        train_data = Dataset(np.random.choice(train_files, config.epoch_size*config.batch_size, replace=True))
        train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=8)
      
        # training
        train_logger = model.learn(train_loader)
        print('Start validation')
        
        # validation
        valid_logger = evaluate_ml(config, model) 
        train_logger[config.eval_metric] = valid_logger[config.eval_metric]
        train_logger['Epoch'] = epoch
        print(train_logger)
        
        if not config.debug:
            with open(log_file, 'a') as csvfile: 
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow([train_logger[k] for k in logger_keys]) 
   
        num_bad_epochs = model.update_lr(valid_logger[config.eval_metric])    
            
        if num_bad_epochs == 0:
            print('Best model saved')
            if not config.debug:
                save_model(config, save_path, model)
        elif num_bad_epochs == config.patience:
            print("%d epochs without improvement, decreasing learning rate"%config.patience)
        elif num_bad_epochs == config.early_stop:
            print("%d epochs without improvement, early stopping"%(config.early_stop))
            break    

        train_logger = {} 
        for k in logger_keys:
            train_logger[k] = []    
            
if __name__ == "__main__":
    start = time.time()
    config = get_config()
    fix_seed(config.seed)
    train(config)
    end = time.time()
    print(config)
    print('Total training time', end-start)
import ecole
import torch
import heapq
import glob
import numpy as np
import argparse
import os
import warnings
import csv
import torch_geometric
import pickle
import gzip
import random
from collections import defaultdict
from brancher.models import create_model, save_model
from common.utils import fix_seed, get_config, sgm
from common.multiprocessing import AgentPool
from common.env import create_env
import time

def update_buffer(buffer, trajectory, key, reward, length_limit=1):
    sub_buffer = buffer[key]
    if len(sub_buffer) < length_limit:
        sub_buffer.append((reward, trajectory))
        sub_buffer.sort(key=lambda x: x[0])
    elif sub_buffer[0][0]<reward:
        # update buffer
        sub_buffer.pop(0)
        sub_buffer.append((reward, trajectory))
        sub_buffer.sort(key=lambda x: x[0])

def buffer2dataset(buffer):
    graphs = []
    next_graphs = []
    rl_infos = []
    for key, sub_buffer in buffer.items():
        for item in sub_buffer:
            if len(item[1])<=0:
                continue
            # draw at most 100 samples from the trajectory
            trajectory = random.sample(item[1], k=min(100, len(item[1])))
            graphs.extend([sample[0] for sample in trajectory])
            next_graphs.extend([sample[1] for sample in trajectory])
            rl_infos.extend([sample[2] for sample in trajectory])
        
    return (graphs, next_graphs, rl_infos) 

def rollout_mp(config, model, buffer, instances, rng, mode='train'):
    assert mode in ['train', 'warmup', 'valid']
    
    model.start_eval()
    if mode=='train':
        instances = rng.choice(instances, config.epoch_size)
        agent_pool = AgentPool(model, config.num_job, config.batch_size, 'sample', config.tree, 'ML')
        agent_pool.start()
        t_samples, t_stats, t_queue, t_access = agent_pool.start_job(instances, [config.seed for _ in instances], config, sample_rate=1, length_limit=config.length_limit, block_policy=True)    
        t_access.set()
        t_queue.join()     
        for trajectory, logger in zip(t_samples, t_stats):
            update_buffer(buffer, trajectory, logger['Instance'], logger['Rewards']) 
        t_samples = sum(t_samples, [])
        graphs = [sample[0] for sample in t_samples]
        next_graphs = [sample[1] for sample in t_samples]
        rl_infos = [sample[2] for sample in t_samples]
        nn_list = [result['Nb_nodes'] for result in t_stats]
        time_list = [result['Time'] for result in t_stats]
        rewards_list = [result['Rewards'] for result in t_stats]
    elif mode=='warmup':
        agent_pool = AgentPool(model, config.num_job, config.batch_size, 'greedy', config.tree, 'ML')
        agent_pool.start()
        t_samples, t_stats, t_queue, t_access = agent_pool.start_job(instances, [config.seed for _ in instances], config, sample_rate=1, length_limit=config.length_limit, block_policy=True)    
        t_access.set()
        t_queue.join()     
        for trajectory, logger in zip(t_samples, t_stats):
            update_buffer(buffer, trajectory, logger['Instance'], logger['Rewards']) 
        t_samples = sum(t_samples, [])
        graphs = [sample[0] for sample in t_samples]
        next_graphs = [sample[1] for sample in t_samples]
        rl_infos = [sample[2] for sample in t_samples]
        nn_list = [result['Nb_nodes'] for result in t_stats]
        time_list = [result['Time'] for result in t_stats]
        rewards_list = [result['Rewards'] for result in t_stats]   
    else:
        agent_pool = AgentPool(model, config.num_job, config.batch_size, 'greedy', config.tree, 'ML')
        agent_pool.start()
        v_samples, v_stats, v_queue, v_access = agent_pool.start_job(instances, [config.seed for _ in instances], config, sample_rate=0, block_policy=True) 
        v_access.set()
        v_queue.join()    
        graphs, next_graphs, rl_infos = [], [], []
        nn_list = [result['Nb_nodes'] for result in v_stats]
        time_list = [result['Time'] for result in v_stats]
        rewards_list = [result['Rewards'] for result in v_stats]
    agent_pool.close()
    
    return (graphs, next_graphs, rl_infos), {'Time': sgm(time_list, 10), 'Nb_nodes': sgm(nn_list, 1), 'Rewards': sgm(rewards_list, 1)}

def train(config):

    train_instances = glob.glob(os.path.join(config.instance_dir, config.task_name, 'train_online') + '/*.lp')   
    valid_instances = glob.glob(os.path.join(config.instance_dir, config.task_name, config.eval_level) + '/*.lp')  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    if not os.path.isfile(config.ckpt_path):
        warnings.warn('Candidate checkpoint does not exist')
        pretrained_ckpt = None
    else:
        pretrained_ckpt = config.ckpt_path
    
    buffer = defaultdict(list)
    save_path = os.path.join(config.save_dir, config.task_name, 'trained_models', config.model_name)
    log_path = os.path.join(config.save_dir, config.task_name, 'logs',  config.model_name)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    rng = np.random.RandomState(config.seed)
    
    model, logger_keys, log_file = create_model(config, log_path, device, pretrained_ckpt)            
    
    if not config.debug:
        with open(log_file, 'w') as csvfile: 
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(logger_keys) 

    data, pretrain_logger = rollout_mp(config, model, buffer, train_instances, rng, 'warmup')         
    
    for i in range(10):
        data, rollout_logger = rollout_mp(config, model, buffer, train_instances, rng)
    
    results = []
    for epoch in range(1,config.epochs+1):
        for i in range(5):
            data, rollout_logger = rollout_mp(config, model, buffer, train_instances, rng)
            train_loader = torch_geometric.loader.DataLoader(list(zip(*data)), batch_size=config.batch_size, shuffle=True)
            train_logger = model.learn(train_loader)
            train_logger[config.eval_metric] = rollout_logger[config.eval_metric] 
            train_logger['Epoch'] = epoch
            print(train_logger)
        
        sl_dataset = buffer2dataset(buffer)
        train_loader = torch_geometric.loader.DataLoader(list(zip(*sl_dataset)), batch_size=config.batch_size, shuffle=False)
        priorities, sl_logger = model.sil(train_loader)

        data, valid_logger = rollout_mp(config, model, buffer, valid_instances, rng, 'valid')       
        train_logger[config.eval_metric] = valid_logger[config.eval_metric] 
        print(valid_logger)

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
            model.update_parameters()    
            print("%d epochs without improvement, decreasing learning rate"%config.patience)
        elif num_bad_epochs == config.early_stop:
            print("%d epochs without improvement, early stopping"%(config.early_stop))
            break   
        
if __name__ == "__main__":
    start = time.time()
    config = get_config()
    fix_seed(config.seed)
    train(config)
    end = time.time()
import ecole
import torch
import glob
import numpy as np
import os
import multiprocessing
from brancher.models import create_model
from common.env import create_env
from common.utils import get_config, fix_seed, sgm
from common.multiprocessing import AgentPool
from common.buffer import build_graph
from scipy.stats.mstats import gmean
from collections import defaultdict

def evaluate_heuristic(config, save=False):
    fix_seed(config.seed)
    time_list = []
    nn_list = []
    rewards_list = []
    
    instances = glob.glob(os.path.join(config.instance_dir, config.task_name, config.eval_level) + '/*.lp')
    instances.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    agent_pool = AgentPool(None, config.num_job, config.batch_size, 'greedy', config.tree, 'heuristic')
    agent_pool.start()
    _, v_stats, v_queue, v_access = agent_pool.start_job(instances, [config.seed for _ in instances], config, block_policy=True)      
    v_access.set()
    v_queue.join()
    agent_pool.close()
    nn_list = [result['Nb_nodes'] for result in v_stats]
    time_list = [result['Time'] for result in v_stats]
    rewards_list = [result['Rewards'] for result in v_stats]
    
    if save:
        result_path = os.path.join(config.save_dir, config.task_name, config.eval_level + '_results', config.model_name + str(config.FSB_probability))
        os.makedirs(result_path, exist_ok=True)
        for logger in v_stats:
            log_path = result_path + '/' + logger['Instance'].split('/')[-1] + '.' + str(config.seed) + '.txt'
            with open(log_path, 'w') as f:
                f.write(' '.join([str(logger['Time']), str(logger['Nb_nodes']), str(logger['Rewards'])])) 
    
    return {'Time': sgm(time_list, 10), 'Nb_nodes': sgm(time_list, 1), 'Rewards': sgm(time_list, 1)}
          
def evaluate_scip(config, save=False):
    fix_seed(config.seed)
    instances = glob.glob(os.path.join(config.instance_dir, config.task_name, config.eval_level) + '/*.lp')
    instances.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    agent_pool = AgentPool(None, config.num_job, config.batch_size, 'greedy', config.tree, 'SCIP')
    agent_pool.start()
    _, v_stats, v_queue, v_access = agent_pool.start_job(instances, [config.seed for _ in instances], config, block_policy=True)      
    v_access.set()
    v_queue.join()
    agent_pool.close()
    nn_list = [result['Nb_nodes'] for result in v_stats]
    time_list = [result['Time'] for result in v_stats]
    rewards_list = [result['Rewards'] for result in v_stats]
    
    if save:
        result_path = os.path.join(config.save_dir, config.task_name, config.eval_level + '_results', 'SCIP')
        os.makedirs(result_path, exist_ok=True)
        for logger in v_stats:
            log_path = result_path + '/' + logger['Instance'].split('/')[-1] + '.' + str(config.seed) + '.txt'
            with open(log_path, 'w') as f:
                f.write(' '.join([str(logger['Time']), str(logger['Nb_nodes']), str(logger['Rewards'])])) 
    
    return {'Time': sgm(time_list, 10), 'Nb_nodes': sgm(time_list, 1), 'Rewards': sgm(time_list, 1)}
    
    
def evaluate_ml(config, model, save=False):     
    fix_seed(config.seed)
    model.start_eval()
    
    instances = glob.glob(os.path.join(config.instance_dir, config.task_name, config.eval_level) + '/*.lp')
    instances.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    agent_pool = AgentPool(model, config.num_job, config.batch_size, 'greedy', False, 'ML')
    agent_pool.start()
    _, v_stats, v_queue, v_access = agent_pool.start_job(instances, [config.seed for _ in instances], config, block_policy=True)    
    v_access.set()
    v_queue.join()
    agent_pool.close()
    nn_list = [result['Nb_nodes'] for result in v_stats]
    time_list = [result['Time'] for result in v_stats]
    rewards_list = [result['Rewards'] for result in v_stats]

    if save:
        result_path = os.path.join(config.save_dir, config.task_name, config.eval_level + '_results', config.model_name)
        os.makedirs(result_path, exist_ok=True)
        for logger in v_stats:
            log_path = result_path + '/' + logger['Instance'].split('/')[-1] + '.' + str(config.seed) + '.txt'
            with open(log_path, 'w') as f:
                f.write(' '.join([str(logger['Time']), str(logger['Nb_nodes']), str(logger['Rewards'])]))  
    

    return {'Time': sgm(time_list, 10), 'Nb_nodes': sgm(time_list, 1), 'Rewards': sgm(time_list, 1)}

    
if __name__=='__main__':
    config = get_config()
    print(config)

    if config.model_name == 'SCIP':
        time_result = []
        nn_result = []
        reward_result = []
        seeds = np.arange(5)
        for seed in seeds:
            config.seed = seed
            evaluate_scip(config, True)
        model_name = config.model_name
        
    elif config.model_name == 'heuristic':
        time_result = []
        nn_result = []
        reward_result = []
        seeds = np.arange(5)        
        for seed in seeds:
            config.seed = seed
            evaluate_heuristic(config, True)
        model_name = config.model_name + '_' + str(config.FSB_probability)
    else:  
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _, _ = create_model(config, './', DEVICE)
        model.load_state_dict(torch.load(config.ckpt_path, map_location=DEVICE))   
        
        for seed in range(5):   
            config.seed = seed
            evaluate_ml(config, model, True)  
        model_name = config.model_name
     
    files = glob.glob(os.path.join(config.save_dir, config.task_name, config.eval_level + '_results', model_name) + '/*.txt')
    nodes = defaultdict(list)
    times = defaultdict(list)
    for file in files:
        key = file.split('/')[-1].split('.')[0]
        with open(file, 'r') as f:
            time, node, _ = f.read().split()
        nodes[key].append(float(node))
        times[key].append(float(time))
        
    mean_time = sgm(sum(times.values(), []), 1)
    std_time = (np.mean([np.std(data) for data in times.values()])/mean_time)*100
    print('Time Result:', str(round(mean_time, 2)) + ' $\pm$ ' + str(round(std_time,1))+ '\%')
    mean_node = sgm(sum(nodes.values(), []), 10)
    std_node = (np.mean([np.std(data) for data in nodes.values()])/mean_node)*100
    print('# Nodes Result:', str(int(round(mean_node, 0))) + ' $\pm$ ' + str(round(std_node,1))+ '\%')



import torch
import numpy as np
import random
import argparse

def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.enabled = False

def sgm(data, sh):
    data = np.array(data)
    return np.exp(np.sum(np.log(np.clip(data+sh, a_min=1, a_max=np.inf))/len(data)))-sh       
    
def get_config():
    parser = argparse.ArgumentParser(description='SORREL')
    parser.add_argument("--task_name", type=str, default="1_set_covering", choices=['1_set_covering', '2_independent_set', '3_combinatorial_auction', '4_facility_location', '5_multi_knapsack'])
    parser.add_argument("--dataset_name", type=str, default="FSB_0.05", help="Dataset name")
    parser.add_argument("--eval_level", type=str, default='test', choices=["valid", "test_standard", "test_transfer"])
    parser.add_argument("--model_name", type=str, default='TD3BC', choices=['BC', 'TD3BC', 'PPO', 'SCIP', 'heuristic'])
    parser.add_argument("--epochs", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=0, help="Seed, default: 0")
    parser.add_argument("--patience", type=int, default=3, help="Epochs to decrease learning rate, default: 3")
    parser.add_argument("--early_stop", type=int, default=5, help="Epochs to stop training, default: 5")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
    parser.add_argument("--epoch_size", type=int, default=312, help="Epoch size")
    parser.add_argument("--num_job", type=int, default=20, help="Number of parallel jobs, defualt: 20")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--target_update_period", type=int, default=1, help="Target update period")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=2.5, help="Alpha for TD3BC")
    parser.add_argument("--kappa", type=float, default=0.8, help="coefficient of min child in q value")
    parser.add_argument("--epsilon", type=float, default=0.1, help="clip range in PPO")
    parser.add_argument("--FSB_probability", type=float, default=0.05, help="Expert probability")
    parser.add_argument("--warmup", action='store_true', help='Whether warm up the model')
    parser.add_argument("--tree", action='store_true', help='Whether use the tree MDP')
    parser.add_argument("--double_Q", action='store_true', help='Whether use double Q networks')
    parser.add_argument("--debug", action='store_true', help='Whether in debug mode')
    parser.add_argument("--time_limit", type=float, default=900.0, help="Time limit for evaluation")
    parser.add_argument("--length_limit", type=int, default=np.inf, help="Episode length limit for collection")
    parser.add_argument("--eval_metric", type=str, default='Nb_nodes', choices=['Nb_nodes', 'Rewards'])
    parser.add_argument("--dataset_dir", type=str, default='./dataset/', help="Dataset saving directory")
    parser.add_argument("--instance_dir", type=str, default='./instances/', help="Problem instances directory")
    parser.add_argument("--save_dir", type=str, default='./checkpoints/', help="Saving directory")
    parser.add_argument("--ckpt_path", type=str, default='./best.pt', help="Pretrained checkpoint path")
    args = parser.parse_args()
    return args    
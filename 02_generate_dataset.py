import os
import gzip
import pickle
from pathlib import Path
import glob
import ecole
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import multiprocessing as mp
import shutil
import time
import json
import math
import argparse
from common.env import create_env

def make_samples(in_queue, out_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """

    while True:
        episode, instance, seed, time_limit, length_params, out_dir, FSB_probability, tree = in_queue.get()
        print(f'[w {os.getpid()}] episode {episode}, seed {seed}, processing instance \'{instance}\'...')


        # Note how we can tuple observation functions to return complex state information
        # We can pass custom SCIP parameters easily
        
        observation, action_set, done, info, env = create_env(instance, FSB_probability, time_limit, length_params[1], seed, 'heuristic')

        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })
        sample_counter = 0
        iter_counter = 0
        if tree:
            all_observations = []
            all_action_sets = []
            all_actions = []
            all_scores = []
            is_FSBs = []           
            while not done:
                (scores, scores_are_FSB), node_observation = observation
                action = action_set[scores[action_set].argmax()]     
                all_observations.append(node_observation)
                all_action_sets.append(action_set)
                all_actions.append(action)
                all_scores.append(scores)
                is_FSBs.append(scores_are_FSB)
                observation, action_set, _, done, info = env.step(action)
                iter_counter += 1
                         
            children_ids = info['Children']
            
            if iter_counter>=length_params[0] and children_ids is not None:
                # keep the sample with a certain probability
                sample_rate = math.log(iter_counter, 1.01)/iter_counter
                for i, children in enumerate(children_ids):
                    if np.random.random()<sample_rate:
                        next_observation = []
                        for child, reward in zip(*children):
                            if child is None:
                                # not visited
                                next_observation.append((None, None, reward))
                            else:    
                                next_observation.append((all_observations[child], all_action_sets[child], reward))             

                        next_observation += [(None, None, 0) for i in range(2-len(next_observation))]
                        data = (all_observations[i], all_action_sets[i], next_observation[0][0], next_observation[0][1],\
                                next_observation[1][0], next_observation[1][1], all_actions[i], all_scores[i],\
                                next_observation[0][2], next_observation[1][2], next_observation[0][0] is None,\
                                next_observation[1][0] is None,  is_FSBs[i], len(children[0]))
                        sample_counter += 1
                        filename = f"{out_dir}/sample_{episode}_{sample_counter}.pkl"
                        with gzip.open(filename, "wb") as f:
                            pickle.dump(data, f)

                        out_queue.put({
                                'type': 'sample',
                                'episode': episode,
                                'instance': instance,
                                'seed': seed,
                                'filename': filename,
                            }) 
        else:    
            all_data = []
            while not done:
                (scores, scores_are_FSB), node_observation = observation
                action = action_set[scores[action_set].argmax()]

                next_observation, next_action_set, reward, done, info = env.step(action)

                if done:
                    next_node_observation = node_observation
                    next_action_set = action_set
                else:
                    next_node_observation = next_observation[1]

                all_data.append([node_observation, action_set, next_node_observation, next_action_set, action, scores, reward, done, scores_are_FSB, info['nb_nodes']])
                observation = next_observation
                action_set = next_action_set
                iter_counter += 1
            
            if iter_counter>=length_limit[0]:
                # keep the sample with a certain probability
                sample_rate = math.log(iter_counter, 1.01)/iter_counter
                for data in all_data: 
                    if np.random.random()<sample_rate:
                        sample_counter += 1
                        filename = f"{out_dir}/sample_{episode}_{sample_counter}.pkl"

                        with gzip.open(filename, "wb") as f:
                            pickle.dump(data, f)

                        out_queue.put({
                                'type': 'sample',
                                'episode': episode,
                                'instance': instance,
                                'seed': seed,
                                'filename': filename,
                            })

        print(f"[w {os.getpid()}] episode {episode} done, {sample_counter} samples")

        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })    
    
      
def send_orders(orders_queue, instances, seed, time_limit, length_params, out_dir, FSB_probability, tree):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    length_params : tuple
        Parameters used to control the episode length.
    out_dir: str
        Output directory in which to write samples.
    out_dir: str
        Output directory in which to write samples.
    FSB_probability : float in [0, 1]
        Probability of running the FSB strategy and collecting samples.
    tree: bool
        Whether use the tree MDP.
    """
    rng = np.random.RandomState(seed)

    episode = 0
    while True:
        instance = rng.choice(instances)
        seed = rng.randint(2**32)
        orders_queue.put([episode, instance, seed, time_limit, length_params, out_dir, FSB_probability, tree])
        episode += 1


def collect_samples(instances, out_dir, rng, n_samples, n_jobs, time_limit, length_params, FSB_probability, tree):   
    orders_queue = mp.Queue(maxsize=2*n_jobs)
    answers_queue = mp.SimpleQueue()
    workers = []
    for i in range(n_jobs):
        p = mp.Process(
                target=make_samples,
                args=(orders_queue, answers_queue),
                daemon=True)
        workers.append(p)
        p.start()

    tmp_samples_dir = f'{out_dir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # start dispatcher
    dispatcher = mp.Process(
            target=send_orders,
            args=(orders_queue, instances, rng.randint(2**32), time_limit, length_params, tmp_samples_dir, FSB_probability, tree),
            daemon=True)
    dispatcher.start()

    # record answers and write samples
    buffer = {}
    current_episode = 0
    i = 0
    in_buffer = 0
    while i < n_samples:
        sample = answers_queue.get()

        # add received sample to buffer
        if sample['type'] == 'start':
            buffer[sample['episode']] = []
        else:
            buffer[sample['episode']].append(sample)
            if sample['type'] == 'sample':
                in_buffer += 1

        # if any, write samples from current episode
        while current_episode in buffer and buffer[current_episode]:
            samples_to_write = buffer[current_episode]
            buffer[current_episode] = []

            for sample in samples_to_write:

                # if no more samples here, move to next episode
                if sample['type'] == 'done':
                    del buffer[current_episode]
                    current_episode += 1

                # else write sample
                else:
                    os.rename(sample['filename'], f'{out_dir}/sample_{i+1}.pkl')
                    in_buffer -= 1
                    i += 1
                    print(f"[m {os.getpid()}] {i} / {n_samples} samples written, ep {sample['episode']} ({in_buffer} in buffer).")

                    # early stop dispatcher (hard)
                    if in_buffer + i >= n_samples and dispatcher.is_alive():
                        dispatcher.terminate()
                        print(f"[m {os.getpid()}] dispatcher stopped...")

                    # as soon as enough samples are collected, stop
                    if i == n_samples:
                        buffer = {}
                        break

    # stop all workers (hard)
    for p in workers:
        p.terminate()

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument("--task_name", type=str, default="1_set_covering", choices=['1_set_covering', '2_independent_set', '3_combinatorial_auction', '4_facility_location', '5_multi_knapsack'])
    parser.add_argument("--FSB_probability", type=float, default=0.05, help="FSB probability in demo")
    parser.add_argument("--time_limit", type=float, default=3600.0, help="Time limit for generation")
    parser.add_argument("--length_limit", type=int, default=2000, help="Episode length limit")
    parser.add_argument("--sampling_threshold", type=int, default=100, help="Threshold for sampling")
    parser.add_argument("--num_job", type=int, default=20, help="Number of parallel jobs")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=0, help="Seed, default: 0")
    parser.add_argument("--tree", action='store_true', help='Whether use the tree MDP')
    parser.add_argument("--instance_dir", type=str, default='./instances', help='Instance dir')
    parser.add_argument("--dataset_dir", type=str, default='./datasets', help='Dataset dir')
    args = parser.parse_args()
    
    rng = np.random.RandomState(args.seed)
    start = time.time()
    
    instance_folder = os.path.join(args.instance_dir, args.task_name, 'train_offline')
    dataset_folder = os.path.join(args.dataset_dir, args.task_name, 'FSB_'+str(args.FSB_probability))
    os.makedirs(dataset_folder, exist_ok=True)
    instances = glob.glob(instance_folder + '/*lp')
    
    collect_samples(instances, dataset_folder, rng, args.num_samples, args.num_job, args.time_limit, (args.sampling_threshold, args.length_limit), args.FSB_probability, args.tree)
    end = time.time()
    
    print('Total Time', end-start)
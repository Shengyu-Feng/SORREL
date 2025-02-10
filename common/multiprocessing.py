import ecole
import threading
import queue
import numpy as np
import torch
from collections import namedtuple
from common.env import create_env
from .buffer import build_graph, RLTreeData, RLData
import torch_geometric

class AgentPool():
    """
    Class holding the reference to the agents and the policy sampler.
    Puts jobs in the queue through job sponsors.
    """
    def __init__(self, brain, n_agents, batch_size, sample_mode, tree, method):
        self.jobs_queue = queue.Queue()
        self.policy_queries_queue = queue.Queue()
        self.policy_sampler = PolicySampler("Policy Sampler", brain, batch_size, sample_mode, self.policy_queries_queue)
        self.agents = [Agent(f"Agent {i}", tree, method, self.jobs_queue, self.policy_queries_queue) for i in range(n_agents)]

    def start(self):
        self.policy_sampler.start()
        for agent in self.agents:
            agent.start()

    def close(self):
        # order the episode sampling agents to stop
        for _ in self.agents:
            self.jobs_queue.put(None)
        self.jobs_queue.join()
        # order the policy sampler to stop
        self.policy_queries_queue.put(None)
        self.policy_queries_queue.join()

    def start_job(self, instances, seeds, config, sample_rate=0, length_limit=np.inf, block_policy=False):
        """
        Starts a job.
        A job is a set of tasks. A task consists of an instance that needs to be solved and instructions
        to do so (sample rate, greediness).
        The job queue is loaded with references to the job sponsor, which is in itself a queue specific
        to a job. It is the job sponsor who holds the lists of tasks. The role of the job sponsor is to
        keep track of which tasks have been completed.
        """
        job_sponsor = queue.Queue()
        samples = []
        stats = []

        policy_access = threading.Event()
        if not block_policy:
            policy_access.set()
        for instance, seed in list(zip(instances, seeds)):
            task = {'instance': instance, 'config': config, 'samples': samples, 'stats': stats,
                     'policy_access': policy_access, 'sample_rate': sample_rate, 'length_limit': length_limit, 'seed': seed}
            job_sponsor.put(task)
            self.jobs_queue.put(job_sponsor)

        ret = (samples, stats, job_sponsor)
        if block_policy:
            ret = (*ret, policy_access)

        return ret

    def wait_completion(self):
        # wait for all running episodes to finish
        self.jobs_queue.join()


class PolicySampler(threading.Thread):
    """
    Gathers policy sampling requests from the agents, and process them in a batch.
    """
    def __init__(self, name, brain, batch_size, sample_mode, requests_queue):
        super().__init__(name=name)
        self.brain = brain
        self.batch_size = batch_size
        self.sample_mode = sample_mode
        self.requests_queue = requests_queue

    def run(self):
        stop_order_received = False
        while True:
            requests = []
            request = self.requests_queue.get()
            while True:
                # check for a stopping order
                if request is None:
                    self.requests_queue.task_done()
                    stop_order_received = True
                    break
                # add request to the batch
                requests.append(request)
                # keep collecting more requests if available, without waiting
                try:
                    request = self.requests_queue.get(block=False)
                except queue.Empty:
                    break
               
            states = [r['state'] for r in requests]
            receivers = [r['receiver'] for r in requests]

            states_loader = torch_geometric.data.DataLoader(states, batch_size=self.batch_size, shuffle=False)
            # process all requests in a batch
            action_idxs = []
            log_probs = []
            for states in states_loader:
                action, log_prob = self.brain.select_action(states.to(self.brain.device), self.sample_mode)
                action_idxs.append(action)
                log_probs.append(log_prob)
            if len(action_idxs)>0:
                action_idxs = torch.cat(action_idxs)
                log_probs = torch.cat(log_probs)
            for action_idx, log_prob, receiver in zip(action_idxs, log_probs, receivers):
                receiver.put((action_idx, log_prob))
                self.requests_queue.task_done()

            if stop_order_received:
                break


class Agent(threading.Thread):
    """
    Agent class. Receives tasks from the job sponsor, runs them and samples transitions if
    requested.
    """
    def __init__(self, name, tree, method, jobs_queue, policy_queries_queue):
        super().__init__(name=name)
        self.tree = tree
        self.method = method
        self.jobs_queue = jobs_queue
        self.policy_queries_queue = policy_queries_queue
        self.policy_answers_queue = queue.Queue()
        

    def run(self):
        while True:
            job_sponsor = self.jobs_queue.get()

            # check for a stopping order
            if job_sponsor is None:
                self.jobs_queue.task_done()
                break

            # Get task from job sponsor
            task = job_sponsor.get()
            instance = task['instance']
            policy_access = task['policy_access']
            config = task['config']
            seed = task['seed']
            samples = task['samples']
            stats = task['stats']
            sample_rate = task['sample_rate']
            length_limit = task['length_limit']

            rng = np.random.RandomState(seed)

            observation, action_set, done, info, env = create_env(instance, config.FSB_probability, config.time_limit, length_limit, seed, self.method)
            all_observations = []
            all_action_sets = []
            all_actions = []
            all_log_probs = []
            all_nb_nodes = []
            rewards = []
            # Run episode
            policy_access.wait()
            iter_count = 0
            
            if self.method == 'SCIP':
                _, _, reward, _, info = env.step({})
                time = info['Time']
                nb_nodes = info['Nb_nodes']
                logger = {'Time': time, 'Nb_nodes': nb_nodes, 'Rewards': -nb_nodes, 'Instance': instance}
                samples.append([])
                stats.append(logger)
                job_sponsor.task_done()
                self.jobs_queue.task_done()
                continue
            
            while not done:
                if self.method == 'ML':
                    with torch.no_grad():
                        graph, candidates = build_graph(observation, action_set)
                        self.policy_queries_queue.put({'state': graph, 'receiver': self.policy_answers_queue})
                        action_idx, log_prob = self.policy_answers_queue.get()
                        action = candidates[action_idx].item() 
                        node_observation = observation
                else:
                    (scores, scores_are_expert), node_observation = observation
                    action = action_set[scores[action_set].argmax()]
                    log_prob = 0
                all_observations.append(node_observation)
                all_action_sets.append(action_set)
                all_actions.append(action)
                all_log_probs.append(log_prob)
                observation, action_set, reward, done, info = env.step(action)
                rewards.append(reward)
                all_nb_nodes.append(info['Nb_nodes'])
                iter_count += 1

            nb_nodes = info["Nb_nodes"]
            time = info["Time"] 
            logger = {'Time': time, 'Nb_nodes': nb_nodes, 'Rewards': -nb_nodes, 'Instance': instance}  
            transitions = [] 
            children_ids = info['Children']
            if len(children_ids)>0:
                def get_return(node):
                    if len(children_ids[node][0])==0:
                        return {node: 0}
                    elif len(children_ids[node][0])==1:                        
                        child = children_ids[node][0][0]
                        if child is None:
                            result = {node: children_ids[node][1][0]}
                        else:
                            result = get_return(child)
                            result[node] = children_ids[node][1][0] + config.gamma*result[child] 
                        return result
                    else:    
                        left_child = children_ids[node][0][0]
                        if left_child is None:
                            left_result = {}
                            left_return = 0
                        else:    
                            left_result = get_return(left_child)
                            left_return = left_result[left_child]
                        right_child = children_ids[node][0][1]
                        if right_child is None:
                            right_result = {}
                            right_return = 0 
                        else:    
                            right_result = get_return(right_child) 
                            right_return = right_result[right_child]

                        coeff = config.kappa if children_ids[node][1][0]<children_ids[node][1][1] else 1-config.kappa
                        
                        node_reward = coeff*(children_ids[node][1][0] + config.gamma*left_return)+\
                                         (1-coeff)*(children_ids[node][1][1] + config.gamma*right_return)
                        left_result.update(right_result)
                        left_result[node] = node_reward
                        return left_result
                
                return_dict = get_return(0) 
                #logger['Rewards'] = return_dict[0]
            
            if sample_rate>0 and len(children_ids)>0:                
                
                for i in range(len(all_observations)):
                    keep_sample = rng.rand() < sample_rate
                    if keep_sample:       
                        if self.tree:
                            graph, action_set  = build_graph(all_observations[i], all_action_sets[i])
                            next_observation = []
                            for child, reward in zip(*children_ids[i]):
                                if child is None:
                                    next_observation.append((None, None, reward))
                                else:
                                    next_observation.append((all_observations[child], all_action_sets[child], reward))

                            next_observation += [(None, None, 0) for i in range(2-len(next_observation))]
                            constraint_feat_dim = all_observations[i].row_features.shape[-1]
                            variable_feat_dim = all_observations[i].column_features.shape[-1]
                            next_graph1, sample_action_set1 = build_graph(None, next_observation[0][1], constraint_feat_dim=constraint_feat_dim, variable_feat_dim=variable_feat_dim)     
                            next_graph2, sample_action_set2 = build_graph(None, next_observation[1][1], constraint_feat_dim=constraint_feat_dim, variable_feat_dim=variable_feat_dim)
                            sample_reward1 = next_observation[0][-1]
                            sample_reward2 = next_observation[1][-1]
                            scores_are_expert = 0
                            done1 = next_observation[0][0] is None
                            done2 = next_observation[1][0] is None
                            nb_nodes = all_nb_nodes[i]
                            candidate_scores = torch.zeros(1)
                            candidate_choice = torch.LongTensor(np.where(all_action_sets[i] == all_actions[i])[0])
                            log_prob = all_log_probs[i]

                            rl_info = RLTreeData(candidate_choice, candidate_scores, torch.FloatTensor([sample_reward1]),\
                                                 torch.FloatTensor([sample_reward2]), torch.LongTensor([done1]),\
                                                 torch.LongTensor([done2]), torch.LongTensor([scores_are_expert]),\
                                                 torch.FloatTensor([nb_nodes]), torch.FloatTensor([return_dict[i]]), torch.FloatTensor([log_prob]))
                            transitions.append((graph, [next_graph1, next_graph2], rl_info))
                        else:
                            graph, _  = build_graph(all_observations[i], all_action_sets[i])
                            if i==len(all_observations)-1:
                                next_graph, _ = build_graph(None, None)
                                done = 1
                            else:
                                constraint_feat_dim = all_observations[i].row_features.shape[-1]
                                variable_feat_dim = all_observations[i].column_features.shape[-1]
                                next_graph, _ = build_graph(all_observations[i+1], all_action_sets[i])
                                done = 0
                            scores_are_expert = 0
                            nb_nodes = all_nb_nodes[i]
                            candidate_scores = torch.zeros(1)
                            reward = all_rewards[i]
                            candidate_choice = torch.LongTensor(np.where(all_action_sets[i] == all_actions[i])[0])
                            log_prob = all_log_probs[i]
                            rl_info = RLData(candidate_choice, candidate_scores, torch.FloatTensor([reward]),\
                                                 torch.LongTensor([done]),torch.LongTensor([scores_are_expert]),\
                                                 torch.FloatTensor([nb_nodes]), torch.FloatTensor([log_prob]))    
                            transitions.append((graph, next_graph, rl_info))   
            
            # record episode samples and stats
            samples.append(transitions)
            stats.append(logger)

            # tell both the agent pool and the original task sponsor that the task is done
            job_sponsor.task_done()
            self.jobs_queue.task_done()


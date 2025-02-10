import os
import numpy as np
import random
import copy
import torch
import torch.nn as nn
from .networks import GNNPolicy
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical
from torch.func import functional_call, stack_module_state
import torch.multiprocessing as mp
from collections import deque


class Scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    Inherits from pytorch's ReduceLROnPlateau scheduler.
    The behavior is the same, except that the num_bad_epochs attribute is **not** reset to
    zero whenever the learning rate is reduced. This means that it will only be reset
    to zero when an improvement on the tracked metric is reported.
    """
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch =+1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs == self.patience:
            self._reduce_lr(self.last_epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

# Actor for policy gradient
class BC():
    def __init__(self, device="cpu", lr=1e-3, tree=False, *args, **kwargs):
        self.device = device   
        self.tree = tree
        self.actor = GNNPolicy().to(self.device)   
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=lr)
        self.train_steps = 0 
        self.r_max = 1
     
    def init_scheduler(self, mode='min', patience=3):
        self.actor_scheduler = Scheduler(self.actor_optimizer, mode=mode, patience=patience, factor=0.2, verbose=True, threshold=1e-6)
    
    def update_lr(self, valid_metric):
        self.actor_scheduler.step(valid_metric)
        return self.actor_scheduler.num_bad_epochs
    
    def state_dict(self):
        return {'actor': self.actor.state_dict(),
               'r_max': self.r_max}
        
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor.to(self.device)
    
    def start_eval(self):
        self.actor.eval()
    
    @torch.no_grad()
    def select_action(self, states, mode='greedy'):
        self.actor.eval()
        logits = self.actor(states)

        if not isinstance(states.nb_candidates, int):
            logits = logits.split(states.nb_candidates.tolist())
            logits = pad_sequence(logits, batch_first=True, padding_value=-torch.inf) 

        if mode=='greedy':
            actions = logits.argmax(dim=-1).cpu()
            log_probs = torch.zeros_like(actions)
        elif mode=='sample': 
            m = Categorical(torch.softmax(logits, dim=-1))
            actions = m.sample()
            log_probs = m.log_prob(actions)
            actions = actions.cpu()
            log_probs = log_probs.cpu()        
        else:
            raise NotImplementedError
        return actions, log_probs
    
    def evaluate(self, valid_loader):
        self.actor.eval()
        acc_list = []
        with torch.no_grad():
            for batch in valid_loader:
                states = batch[0].to(self.device)
                rl_info = batch[-1].to(self.device)
                states, next_states, rl_info = batch
                logits = self.actor.action_logits(states)
                true_scores = pad_sequence(torch.split(rl_info.candidate_scores, states.nb_candidates.tolist()), batch_first=True, padding_value=-1e8)
                true_bestscore = true_scores.max(dim=-1, keepdims=True).values
                predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
                acc_list.append((true_scores.gather(-1, predicted_bestindex) == true_bestscore).float())

        acc_list = torch.cat(acc_list)       
        return {'Val_Accuracy': acc_list.mean().item()}
               
    def prenorm(self, prenorm_loader):
        self.actor.prenorm_init()
        i = 0
        rewards = []
        while True:
            for batch in prenorm_loader:
                states = batch[0].to(self.device)
                rl_info = batch[-1]
                if i==0:
                    if self.tree:
                        rewards.append(rl_info.rewards1)
                        rewards.append(rl_info.rewards2)
                    else:
                        rewards.append(rl_info.rewards)                
                if not self.actor.prenorm(states):
                    break

            if self.actor.prenorm_next() is None:
                break
            i += 1
        rewards = torch.cat(rewards)
        self.r_max = torch.max(rewards).item()
        return i

    def learn(self, train_loader):
        self.actor.train()
        loss_list = []
        acc_list = []
        for batch in train_loader:
            states = batch[0].to(self.device)
            rl_info = batch[-1].to(self.device)
        
            logits = self.actor.action_logits(states)
            loss = F.cross_entropy(logits, rl_info.actions)
            true_scores = pad_sequence(torch.split(rl_info.candidate_scores, states.nb_candidates.tolist()), batch_first=True, padding_value=-1e8)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values
            predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
            acc = ((true_scores.gather(-1, predicted_bestindex) == true_bestscore).float())
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            loss_list.append(loss.item())
            acc_list.append(acc)
            
        acc_list = torch.cat(acc_list)
        
        return {'Loss': np.mean(loss_list), 'Accuracy': acc_list.mean().item()}


class AC(BC):
    def __init__(self, device="cpu", lr=1e-3, tree=False, gamma=0.99, kappa=0.5, target_update_period=1, double_Q=False, *args, **kwargs):
        super().__init__(device=device, lr=lr, tree=tree, *args, **kwargs)        
        
        self.tau = 1e-3
        self.gamma = gamma
        self.tree = tree
        self.target_update_period = target_update_period
        self.critic_criterion = nn.HuberLoss()            
        self.critic = GNNPolicy().to(self.device)   
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=lr)
        self.critic_target = copy.deepcopy(self.critic) 
        self.double_Q = double_Q
        self.kappa = kappa
        if self.double_Q:
            self.critic_p = GNNPolicy().to(self.device)
            self.critic_p_target = copy.deepcopy(self.critic_p)
            self.critic_optimizer = optim.Adam(params=list(self.critic.parameters())+list(self.critic_p.parameters()), lr=lr) 
        else: 
            self.critic_p = None
            self.critic_p_target = None
    
    def init_scheduler(self, mode='min', patience=3):
        self.critic_scheduler = Scheduler(self.critic_optimizer, mode=mode, patience=patience, factor=0.2, verbose=True)
        self.actor_scheduler = Scheduler(self.actor_optimizer, mode=mode, patience=patience, factor=0.2, verbose=True)
    
    def update_lr(self, valid_metric):
        self.critic_scheduler.step(valid_metric)
        self.actor_scheduler.step(valid_metric)
        return self.actor_scheduler.num_bad_epochs
    
    def state_dict(self):
        return {'actor': self.actor.state_dict(), 
                'critic': self.critic.state_dict(),
               'critic_p': self.critic_p.state_dict() if self.double_Q else None,
               'r_max': self.r_max}
        
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor.to(self.device)
        if 'critic' in state_dict:
            self.critic.load_state_dict(state_dict['critic'])
            self.critic.to(self.device)
            self.critic_target = copy.deepcopy(self.critic)  
        if self.double_Q:
            self.critic_p.load_state_dict(state_dict['critic_p'])
            self.critic_p.to(self.device)
            self.critic_p_target = copy.deepcopy(self.critic_p) 
        if 'r_max' in state_dict:
            self.r_max = state_dict['r_max']
    
    def warmup(self, warmup_loader):
        self.warmup_network(self.actor, warmup_loader)
    
    def Q_targets(self, critic, critic_target, critic_p, critic_p_target, actor, next_states, rewards, dones):
        logits = actor.action_logits(next_states)  
        action_probs = torch.softmax(logits, dim=-1)
        Q_targets_next = self.critic_scores(critic_target, next_states).detach() 
        if self.double_Q:
            Q_p_targets_next = self.critic_scores(critic_p_target, next_states).detach() 
            Q_targets_next = torch.min(Q_targets_next, Q_p_targets_next)
        Q_targets_next.masked_fill_(Q_targets_next == -float('inf'), -1e6)           
        Q_targets_next = (action_probs*Q_targets_next).sum(1, keepdim=True).detach()
        Q_targets = rewards.unsqueeze(-1)/self.r_max + (self.gamma * Q_targets_next * (1-dones.unsqueeze(-1)))
        return Q_targets
    
    def update_critic(self, critic, critic_target, critic_p, critic_p_target, actor, optimizer, states, next_states, rl_info):
        critic.train()
        if self.double_Q:
            self.critic_p.train()
        with torch.no_grad():
            if self.tree:
                Q_targets1 = self.Q_targets(critic, critic_target, critic_p, critic_p_target, actor, next_states[0], rl_info.rewards1, rl_info.dones1)
                Q_targets2 = self.Q_targets(critic, critic_target, critic_p, critic_p_target, actor, next_states[1], rl_info.rewards2, rl_info.dones2)
                coeff = torch.where(rl_info.rewards1<rl_info.rewards2, self.kappa, 1-self.kappa).unsqueeze(-1)
                Q_targets = coeff*Q_targets1 + (1-coeff)*Q_targets2
            else:
                Q_targets = self.Q_targets(critic, critic_target, critic_p, critic_p_target, actor, next_states, rl_info.rewards, rl_info.dones)
               

        Q_a_s = self.critic_scores(self.critic, states)
        Q_expected = Q_a_s.gather(1, rl_info.actions.unsqueeze(1))
        critic_loss = self.critic_criterion(Q_expected, Q_targets) 
        
        if self.double_Q:
            Q_p_a_s = self.critic_scores(self.critic_p, states)
            Q_p_expected = Q_p_a_s.gather(1, rl_info.actions.unsqueeze(1))
            critic_p_loss = self.critic_criterion(Q_p_expected, Q_targets) 
            critic_loss = critic_loss + critic_p_loss
                        
        optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(critic.parameters(), 1)
        if self.double_Q:
            clip_grad_norm_(critic_p.parameters(), 1)
        optimizer.step()
        
        return critic_loss.item()    
    
    def update_actor(self, actor, critic, critic_p, optimizer, states, rl_info):
        actor.train()
        logits = actor.action_logits(states)
        with torch.no_grad():
            Q = self.critic_scores(critic, states).detach() 
            if self.double_Q:
                Q_p = self.critic_scores(critic_p, states).detach()            
                Q = torch.min(Q, Q_p)
            Q.masked_fill_(Q== -float('inf'), -1e6) 
            
        action_probs = torch.softmax(logits, dim=-1)    
        actor_loss = -(action_probs * Q).sum(1).mean()
        
        optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(actor.parameters(), 1)
        optimizer.step()
        
        return actor_loss.item()   
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data) 
    
    def critic_scores(self, net, states):
        logits = net.action_logits(states)
        return logits
    
    def learn(self, train_loader):
        actor_loss_list = []
        critic_loss_list = []
        for batch in train_loader:
            states, next_states, rl_info = batch
            states = states.to(self.device)
            rl_info = batch[-1].to(self.device)  

            if self.tree:
                next_states = [data.to(self.device) for data in next_states] 
            else:
                next_states = next_states.to(self.device)
                
            critic_loss =  self.update_critic(self.critic, self.critic_target, self.critic_p, self.critic_p_target, self.actor, self.critic_optimizer, states, next_states, rl_info)
            critic_loss_list.append(critic_loss)
            self.train_steps += 1
            # ------------------- update target network ------------------- #
            if self.train_steps % self.target_update_period ==0:
                actor_loss = self.update_actor(self.actor, self.critic_target, self.critic_p, self.actor_optimizer, states, rl_info)
                actor_loss_list.append(actor_loss)
                # follow from original repo
                self.soft_update(self.critic, self.critic_target)
                if self.double_Q:
                    self.soft_update(self.critic_p, self.critic_p_target)
                    
        return {'Critic loss': np.mean(critic_loss_list),
               'Actor loss': np.mean(actor_loss_list) if len(actor_loss_list)>0 else 0}

         
# TD3BC: offline pretraining         
class TD3BC(AC):
    def __init__(self, device="cpu", lr=1e-3, tree=False, gamma=0.99, kappa=0.5, target_update_period=1, double_Q=False, alpha=2.5, *args, **kwargs):
        super().__init__(device=device, lr=lr, tree=tree, gamma=gamma, kappa=kappa, target_update_period=target_update_period, double_Q=double_Q, *args, **kwargs) 
        self.alpha = alpha
    
    def update_actor(self, actor, critic, critic_p, optimizer, states, rl_info):  
        actor.train()
        logits = actor.action_logits(states)
        with torch.no_grad():
            Q = self.critic_scores(critic, states).detach() 
            if self.double_Q:
                Q_p = self.critic_scores(critic_p, states).detach()            
                Q = torch.min(Q, Q_p)
            Q.masked_fill_(Q== -float('inf'), -1e6) 
            
        lmbda = self.alpha/Q.gather(1, rl_info.actions.unsqueeze(-1)).abs().mean().detach()    
            
        action_probs = torch.softmax(logits, dim=-1)    
        actor_loss = -lmbda*(action_probs * Q).sum(1).mean() + F.cross_entropy(logits, rl_info.actions)
        
        optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(actor.parameters(), 1)
        optimizer.step()        
        return actor_loss.item()  

# PPO: online finetuning         
class PPO(AC):
    def __init__(self, device="cpu", lr=1e-3, tree=False, gamma=0.99, epsilon=0.1, target_update_period=1, double_Q=False, pretrained_checkpoint=None, *args, **kwargs):
        super().__init__(device=device, lr=lr, tree=tree, gamma=gamma, target_update_period=target_update_period *args, **kwargs) 
        
        if pretrained_checkpoint is not None:
            self.load_state_dict(torch.load(pretrained_checkpoint))
        self.clip_epsilon = epsilon
        
    def critic_scores(self, net, states):
        logits = net.action_logits(states)
        return logits.max(1)[0]
        
    
    @torch.no_grad()
    def compute_advantage(self, data_loader):
        all_advantages = []
        for batch in data_loader:
            states, next_states, rl_info = batch
            states = states.to(self.device)
            rl_info = rl_info.to(self.device)
            value = self.critic_scores(self.critic, states).detach()
            advantages = rl_info.returns/self.r_max - value  
            all_advantages.append(advantages.detach().cpu())
            
        all_advantages_tensor = torch.cat(all_advantages)
        adv_mean, adv_std = all_advantages_tensor.mean(), all_advantages_tensor.std()+1e-6
        all_advantages = [(advantages-adv_mean)/adv_std for advantages in all_advantages] 
        
        return all_advantages
    
    def sil(self, train_loader):
        self.actor.train()
        loss_list = []
        acc_list = []
        priority_list = []
        for batch in train_loader:
            states, next_states, rl_info = batch
            states = states.to(self.device)
            rl_info = rl_info.to(self.device)
            logits = self.actor.action_logits(states)
            value = self.critic_scores(self.critic, states)
            weight = torch.clip(rl_info.returns/self.r_max-value, min=0)
            priority_list.append((weight*self.r_max).detach().cpu().numpy())
            correct = (logits.argmax(-1)==rl_info.actions)
            actor_loss = F.cross_entropy(logits, rl_info.actions, reduction='none')
            actor_loss = (actor_loss*weight.detach()*10).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            critic_loss = 0.5*weight.pow(2).mean()/10
            critic_loss.backward()
            self.critic_optimizer.step()
            loss_list.append(actor_loss.item()+critic_loss.item())
            acc_list.append(correct.float())
        acc_list = torch.cat(acc_list)
        priority_list = np.concatenate(priority_list)
        return priority_list, {'Loss': np.mean(loss_list), 'Acc': torch.mean(acc_list).item()}
    
    def learn(self, train_loader):
        self.actor.train()
        self.critic.train()
        actor_loss_list = []
        critic_loss_list = []
        all_advantages = self.compute_advantage(train_loader)
        n_samples = len(train_loader.dataset)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            states, next_states, rl_info = batch
            states = states.to(self.device)
            rl_info = batch[-1].to(self.device)
            advantages = all_advantages[i].to(self.device)
            value = self.critic_scores(self.critic, states)
            error = rl_info.returns/self.r_max - value
            critic_loss = 0.5*error.pow(2).sum()/n_samples
            critic_loss.backward()
            critic_loss_list.append(critic_loss.item())

            logits = self.actor.action_logits(states)

            m = Categorical(torch.softmax(logits, dim=-1))
            log_probs = m.log_prob(rl_info.actions)
            old_probs = rl_info.log_probs
            ratios = torch.exp(log_probs - old_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            actor_loss = (-torch.min(surr1, surr2)).sum()/n_samples
            actor_loss.backward()
            actor_loss_list.append(actor_loss.item())

        clip_grad_norm_(self.actor.parameters(), 1)     
        clip_grad_norm_(self.critic.parameters(), 1) 
        self.actor_optimizer.step()    
        self.critic_optimizer.step()  
        return {
                'Critic loss': np.mean(critic_loss_list),
                'Actor loss': np.mean(actor_loss_list),
               }    
    
     
def create_model(config, log_path, device, pretrained_checkpoint=None):
    if config.model_name == 'BC':
        model = BC(device=device, lr=config.lr, tree=config.tree)
        model.init_scheduler(mode='min' if config.eval_metric=='Nb_nodes' else 'max', patience=config.patience)
        logger_keys = ['Epoch', 'Loss', 'Accuracy', config.eval_metric]   
        log_file = os.path.join(log_path, '_'.join([config.model_name, str(config.seed)])+".csv")
    elif config.model_name == 'TD3BC':     
        model = TD3BC(device=device, lr=config.lr, double_Q=config.double_Q, tree=config.tree, target_update_period=config.target_update_period, gamma=config.gamma, kappa=config.kappa, alpha=config.alpha)
        model.init_scheduler(mode='min' if config.eval_metric=='Nb_nodes' else 'max', patience=config.patience)
        logger_keys = ['Epoch', 'Actor loss', 'Critic loss', config.eval_metric]    
        log_file = os.path.join(log_path, '_'.join([config.model_name, str(config.alpha), str(config.seed)])+".csv")      
    elif config.model_name == 'PPO':  
        model = PPO(device=device, lr=config.lr, tree=config.tree, target_update_period=config.target_update_period, gamma=config.gamma, epsilon=config.epsilon, pretrained_checkpoint=pretrained_checkpoint)
        model.init_scheduler(mode='min' if config.eval_metric=='Nb_nodes' else 'max', patience=config.patience)
        logger_keys = ['Epoch', 'Actor loss', 'Critic loss', config.eval_metric]
        log_file = os.path.join(log_path, '_'.join([config.model_name, str(config.seed)])+".csv")
    else:
        raise NotImplementedError
        
    return model, logger_keys, log_file 

def save_model(config, save_dir, model):
    if config.model_name == 'BC':
        torch.save(model.state_dict(), os.path.join(save_dir, '_'.join([config.model_name, str(config.seed)]) +".pth"))
    elif config.model_name == 'TD3BC':
        torch.save(model.state_dict(), os.path.join(save_dir, '_'.join([config.model_name, str(config.alpha), str(config.kappa), str(config.gamma), str(int(config.double_Q)), str(config.seed)]) +".pth"))
    elif config.model_name == 'PPO':
        torch.save(model.state_dict(), os.path.join(save_dir, '_'.join([config.model_name, str(config.seed)]) +".pth"))
    else:
        raise NotImplementedError
import torch_geometric
import numpy as np
import gzip
import pickle
import torch

class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        candidates,
        nb_candidates,
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        
    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class RLData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        candidate_choice,
        candidate_scores,
        reward,
        done,
        is_expert,
        nb_nodes,
        log_probs=torch.zeros(1),
    ):
        super().__init__()
        
        self.actions = candidate_choice
        self.candidate_scores = candidate_scores
        self.rewards = reward   
        self.dones = done
        self.is_expert = is_expert
        self.nb_nodes = nb_nodes
        self.log_probs = log_probs
        
class RLTreeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        candidate_choice,
        candidate_scores,
        reward1,
        reward2,
        done1,
        done2,
        is_expert,
        nb_nodes,
        returns=0,
        log_probs=0
    ):
        super().__init__()
        
        self.actions = candidate_choice
        self.candidate_scores = candidate_scores
        self.returns = returns
        self.rewards1 = reward1
        self.rewards2 = reward2
        self.dones1 = done1
        self.dones2 = done2
        self.is_expert = is_expert
        self.nb_nodes = nb_nodes  
        self.log_probs = log_probs

        
    
def build_graph(sample_observation, action_sets, constraint_feat_dim=5, edge_feat_dim=1, variable_feat_dim=19):
    if sample_observation is None:
        candidates = torch.zeros(1).long()
        graph = BipartiteNodeData(
            torch.zeros((1,constraint_feat_dim), dtype=torch.float),
            torch.empty((2,0), dtype=torch.long),
            torch.empty((0,edge_feat_dim), dtype=torch.float),
            torch.zeros((1,variable_feat_dim), dtype=torch.float),
            torch.zeros((1,), dtype=torch.long), 
            len(candidates)
        )   
        graph.num_nodes = 2
    else:
        candidates = torch.LongTensor(action_sets.astype(np.int32))
        constraint_features = sample_observation.row_features
        edge_indices = sample_observation.edge_features.indices.astype(np.int32)
        edge_features = np.expand_dims(sample_observation.edge_features.values, axis=-1)
        variable_features = sample_observation.column_features
        num_variables = len(variable_features)
        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
            torch.LongTensor(candidates), 
            len(candidates)
        )
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]      
    return graph, candidates
        
class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], "rb") as f:
            sample = pickle.load(f)

        sample_observation, sample_action_set, next_sample_observation, next_sample_action_set, sample_action, sample_scores, sample_reward, done, scores_are_expert, nb_nodes = sample
        # We note on which variables we were allowed to branch, the scores as well as the choice
        # taken by strong branching (relative to the candidates)
        constraint_feat_dim = sample_observation.row_features.shape[-1]
        variable_feat_dim = sample_observation.column_features.shape[-1]
        
        graph, sample_action_set = build_graph(sample_observation, sample_action_set)
        next_graph, _ = build_graph(next_sample_observation, next_sample_action_set, constraint_feat_dim=constraint_feat_dim, variable_feat_dim=variable_feat_dim)     
        sample_scores = torch.FloatTensor(sample_scores)
        candidate_scores = sample_scores[sample_action_set]
        candidate_choice = torch.where(sample_action_set == sample_action)[0][0]
        
        rl_info = RLData(candidate_choice, candidate_scores, torch.FloatTensor([sample_reward]), torch.LongTensor([done]), torch.LongTensor([scores_are_expert]), torch.FloatTensor([nb_nodes]))
        return graph, next_graph, rl_info

    
class TreeDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], "rb") as f:
            sample = pickle.load(f)

        sample_observation, sample_action_set, next_sample_observation1, next_sample_action_set1, next_sample_observation2, next_sample_action_set2, sample_action, sample_scores, sample_reward1, sample_reward2, done1, done2, scores_are_expert, nb_nodes = sample
        # We note on which variables we were allowed to branch, the scores as well as the choice
        # taken by strong branching (relative to the candidates)
        constraint_feat_dim = sample_observation.row_features.shape[-1]
        variable_feat_dim = sample_observation.column_features.shape[-1]
        
        graph, sample_action_set = build_graph(sample_observation, sample_action_set)
        next_graph1, sample_action_set1 = build_graph(next_sample_observation1, next_sample_action_set1, constraint_feat_dim=constraint_feat_dim, variable_feat_dim=variable_feat_dim)     
        next_graph2, sample_action_set2 = build_graph(next_sample_observation2, next_sample_action_set2, constraint_feat_dim=constraint_feat_dim, variable_feat_dim=variable_feat_dim)     
        sample_scores = torch.FloatTensor(sample_scores)
        candidate_scores = sample_scores[sample_action_set]
        candidate_choice = torch.where(sample_action_set == sample_action)[0][0]
        
        rl_info = RLTreeData(candidate_choice, candidate_scores, torch.FloatTensor([sample_reward1]), torch.FloatTensor([sample_reward2]), torch.LongTensor([done1]), torch.LongTensor([done2]), torch.LongTensor([scores_are_expert]), torch.FloatTensor([nb_nodes]))
        
             
        return graph, [next_graph1, next_graph2], rl_info    
    
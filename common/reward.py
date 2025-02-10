import ecole
import pyscipopt
from .search_tree import SearchTree

class Dumb:
    def __init__(self):
        self.last_db = 0

    def before_reset(self, model):
        pass

    def extract(self, model, done):
        return 0

class DualBoundDiff:
    def __init__(self):
        self.last_db = 0

    def before_reset(self, model):
        pass

    def extract(self, model, done):
        # Unconditionally getting reward as reward_funcition.extract may have side effects
        db = model.as_pyscipopt().getDualbound()
        reward = abs(self.last_db - db)
        self.last_db = db
        return reward
        
class DualBound:
    def __init__(self):
        pass

    def before_reset(self, model):
        pass

    def extract(self, model, done):
        # Unconditionally getting reward as reward_funcition.extract may have side effects
        db = model.as_pyscipopt().getDualbound()
        return db


        return reward    
    
class Children:
    def __init__(self, debug=False):
        self.tree = None
        self.debug_mode = debug

    def before_reset(self, model):
        self.tree = None

    def extract(self, model, done):
        if not done:
            if self.tree is None:
                self.tree = SearchTree(model)
            else:
                self.tree.update_tree(model)
                if self.debug_mode:
                    self.tree.render()
            return None
        # instance was pre-solved
        if self.tree is None:
            return []
        #self.tree.update_tree(model)
        node2position = {node: i for i, node in enumerate(self.tree.tree.graph['visited_node_ids'])}

        childrens = []
        for node in self.tree.tree.graph['visited_node_ids']:
            # ignore not visited children
            parent_lower_bound = self.tree.tree.nodes[node]['lower_bound']

            childrens.append(([
                node2position[child] if child in node2position else None for child in self.tree.tree.successors(node)
            ], [
                self.tree.tree.nodes[child]['lower_bound']-parent_lower_bound for child in self.tree.tree.successors(node)
            ]))

        if self.debug_mode:
            print('\nB&B tree:')
            print(f'All nodes saved: {self.tree.tree.nodes()}')
            print(f'Visited nodes: {self.tree.tree.graph["visited_node_ids"]}')
            self.tree.render()
            self.tree.update_tree(model)
            self.tree.render()
            for node in [node for node in self.tree.tree.nodes]:
                if node not in self.tree.tree.graph['visited_node_ids']:
                    self.tree.tree.remove_node(node)
            self.tree.step_idx += 1
            self.tree.render()


        return childrens
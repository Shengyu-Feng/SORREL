import ecole
import os
import argparse
import numpy as np
from tqdm import tqdm
from ecole.instance import SetCoverGenerator, IndependentSetGenerator, CombinatorialAuctionGenerator, CapacitatedFacilityLocationGenerator



def generate_mknapsack(number_of_items, number_of_knapsacks, filename, random,
    min_range=10, max_range=20, scheme='weakly correlated'):
    """
    Generate a Multiple Knapsack problem following a scheme among those found in section 2.1. of
        Fukunaga, Alex S. (2011). A branch-and-bound algorithm for hard multiple knapsack problems.
        Annals of Operations Research (184) 97-119.
    Saves it as a CPLEX LP file.
    Parameters
    ----------
    number_of_items : int
        The number of items.
    number_of_knapsacks : int
        The number of knapsacks.
    filename : str
        Path to the file to save.
    random : numpy.random.RandomState
        A random number generator.
    min_range : int, optional
        The lower range from which to sample the item weights. Default 10.
    max_range : int, optional
        The upper range from which to sample the item weights. Default 20.
    scheme : str, optional
        One of 'uncorrelated', 'weakly correlated', 'strongly corelated', 'subset-sum'. Default 'weakly correlated'.
    """
    weights = random.randint(min_range, max_range, number_of_items)

    if scheme == 'uncorrelated':
        profits = random.randint(min_range, max_range, number_of_items)

    elif scheme == 'weakly correlated':
        profits = np.apply_along_axis(
            lambda x: random.randint(x[0], x[1]),
            axis=0,
            arr=np.vstack([
                np.maximum(weights - (max_range-min_range), 1),
                weights + (max_range-min_range)]))

    elif scheme == 'strongly correlated':
        profits = weights + (max_range - min_range) / 10

    elif scheme == 'subset-sum':
        profits = weights

    else:
        raise NotImplementedError

    capacities = np.zeros(number_of_knapsacks, dtype=int)
    capacities[:-1] = random.randint(0.4 * weights.sum() // number_of_knapsacks,
                                        0.6 * weights.sum() // number_of_knapsacks,
                                        number_of_knapsacks - 1)
    capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

    with open(filename, 'w') as file:
        file.write("maximize\nOBJ:")
        for knapsack in range(number_of_knapsacks):
            for item in range(number_of_items):
                file.write(f" +{profits[item]} x{item+number_of_items*knapsack+1}")

        file.write("\n\nsubject to\n")
        for knapsack in range(number_of_knapsacks):
            variables = "".join([f" +{weights[item]} x{item+number_of_items*knapsack+1}"
                                 for item in range(number_of_items)])
            file.write(f"C{knapsack+1}:" + variables + f" <= {capacities[knapsack]}\n")

        for item in range(number_of_items):
            variables = "".join([f" +1 x{item+number_of_items*knapsack+1}"
                                 for knapsack in range(number_of_knapsacks)])
            file.write(f"C{number_of_knapsacks+item+1}:" + variables + " <= 1\n")

        file.write("\nbinary\n")
        for knapsack in range(number_of_knapsacks):
            for item in range(number_of_items):
                file.write(f" x{item+number_of_items*knapsack+1}")

generator_dict = {
    '1_set_covering': {
            'standard': SetCoverGenerator(n_rows=500),
            'transfer': SetCoverGenerator(n_rows=1000),
        },
    '2_independent_set': {
            'standard': IndependentSetGenerator(n_nodes=500),
            'transfer': IndependentSetGenerator(n_nodes=1000),
        },
    '3_combinatorial_auction': {
            'standard': CombinatorialAuctionGenerator(n_items=100, n_bids=500),
            'transfer': CombinatorialAuctionGenerator(n_items=200, n_bids=1000),
        },           
    '4_facility_location': {
            'standard': CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=100),
            'transfer': CapacitatedFacilityLocationGenerator(n_customers=200, n_facilities=100),
        }   
    }
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Instance')
    parser.add_argument("--task_name", type=str, default="1_set_covering", choices=['1_set_covering', '2_independent_set', '3_combinatorial_auction', '4_facility_location', '5_multi_knapsack'])
    parser.add_argument("--instance_dir", type=str, default='./instances/', help="Problem instances directory")
    parser.add_argument("--num_offline", type=int, default=10000, help="Number of offline training instances")
    parser.add_argument("--num_online", type=int, default=200, help="Number of online training instances")
    parser.add_argument("--num_valid", type=int, default=20, help="Number of valid instances")
    parser.add_argument("--num_standard", type=int, default=100, help="Number of standard testing instance")
    parser.add_argument("--num_transfer", type=int, default=20, help="Number of transfer testing instances")
    args = parser.parse_args()
    
    if args.task_name in ['1_set_covering', '2_independent_set', '3_combinatorial_auction',  '4_facility_location']:
        save_dir = os.path.join(args.instance_dir, args.task_name)
        os.makedirs(save_dir, exist_ok=True)

        standard_generator = generator_dict[args.task_name]['standard']
        transfer_generator = generator_dict[args.task_name]['transfer']

        # generate 10000 train instances
        standard_generator.seed(0)
        offline_dir = os.path.join(save_dir, 'train_offline')
        os.makedirs(offline_dir, exist_ok=True)
        for i in tqdm(range(args.num_offline)):
            instance = next(standard_generator)
            instance.write_problem(os.path.join(offline_dir, 'instance_%d.lp'%i))
            
        online_dir = os.path.join(save_dir, 'train_online')
        os.makedirs(online_dir, exist_ok=True)
        for i in tqdm(range(args.num_online)):
            instance = next(standard_generator)
            instance.write_problem(os.path.join(online_dir, 'instance_%d.lp'%i))
        
        # generate 100 standard test instances
        standard_generator.seed(1)
        standard_dir = os.path.join(save_dir, 'test_standard')
        os.makedirs(standard_dir, exist_ok=True)
        for i in tqdm(range(args.num_standard)):
            instance = next(standard_generator)
            instance.write_problem(os.path.join(standard_dir, 'instance_%d.lp'%i))
        
        # generate 20 valid instances for SORREL
        standard_generator.seed(2)
        valid_dir = os.path.join(save_dir, 'valid')
        os.makedirs(valid_dir, exist_ok=True)
        for i in tqdm(range(args.num_valid)):
            instance = next(standard_generator)
            instance.write_problem(os.path.join(valid_dir, 'instance_%d.lp'%i))  
                  
        # generate 20 transfer test instances
        transfer_generator.seed(1)
        transfer_dir = os.path.join(save_dir, 'test_transfer')
        os.makedirs(transfer_dir, exist_ok=True)
        for i in tqdm(range(args.num_transfer)):
            instance = next(transfer_generator)
            instance.write_problem(os.path.join(transfer_dir, 'instance_%d.lp'%i))
    else:
        # Generating multi knapsack problem instances
        save_dir = os.path.join(args.instance_dir, '5_multi_knapsack')

        rng = np.random.RandomState(0)
        offline_dir = os.path.join(save_dir, 'train_offline')
        os.makedirs(offline_dir, exist_ok=True)
        for i in tqdm(range(args.num_offline)):
            generate_mknapsack(100, 6, os.path.join(offline_dir, 'instance_%d.lp'%i), rng, min_range=10, max_range=20, scheme='subset-sum')

        online_dir = os.path.join(save_dir, 'train_online')
        os.makedirs(online_dir, exist_ok=True)
        for i in tqdm(range(args.num_online)):
            generate_mknapsack(100, 6, os.path.join(online_dir, 'instance_%d.lp'%i), rng, min_range=10, max_range=20, scheme='subset-sum')

        rng = np.random.RandomState(1)
        standard_dir = os.path.join(save_dir, 'test_standard')
        os.makedirs(standard_dir, exist_ok=True)
        for i in tqdm(range(args.num_standard)):
            generate_mknapsack(100, 6, os.path.join(standard_dir, 'instance_%d.lp'%i), rng, min_range=10, max_range=20, scheme='subset-sum')

        rng = np.random.RandomState(2)
        valid_dir = os.path.join(save_dir, 'valid')
        os.makedirs(valid_dir, exist_ok=True)
        for i in tqdm(range(args.num_valid)):
            generate_mknapsack(100, 6, os.path.join(valid_dir, 'instance_%d.lp'%i), rng, min_range=10, max_range=20, scheme='subset-sum')

        rng = np.random.RandomState(1)
        transfer_dir = os.path.join(save_dir, 'test_transfer')
        os.makedirs(transfer_dir, exist_ok=True)
        for i in tqdm(range(args.num_transfer)):
            generate_mknapsack(100, 12, os.path.join(transfer_dir, 'instance_%d.lp'%i), rng, min_range=10, max_range=20, scheme='subset-sum')
    


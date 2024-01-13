from rdkit import Chem
import numpy as np
import ot
import os
import pickle

def main(config):
    train_targets = np.load(configs['opt']['train_prop']).astype('float')
    target_prop = configs['opt']['prop']
    
    if target_prop == 'logp':
        from scorers.logp_scores import score_function
    elif target_prop == 'tpsa':
        from scorers.tpsa_scores import score_function
    elif target_prop == 'qed':
        from scorers.qed_scores import score_function
    elif target_prop == 'sa':
        from scorers.sa_scores import score_function
    elif target_prop == 'multi':
        from scorers.multi_scores import score_function
    elif target_prop == 'mw':
        from scorers.mw_scores import score_function
    elif target_prop == 'aroring':
        from scorers.aroring_scores import score_function
    elif target_prop == 'rotbond':
        from scorers.rotbond_scores import score_function
    else:
        raise ValueError("please define the score function first.")
    result_save_dir = configs['opt']['output']
    with open((os.path.join(result_save_dir, "%s_smiles.pkl" % configs['opt']['prop'])), "rb") as f:
        results_smiles = pickle.load(f)
    results_mols = [Chem.MolFromSmiles(s) for s in results_smiles]
    results_scores = [score_function(m) for m in results_mols]
    # Create cost matrix (here, using the L2 distance between elements)

    cost_matrix = ot.dist(np.array(train_targets).reshape((-1, 1)), np.array(results_scores).reshape((-1, 1)))

    # Calculate Wasserstein distance
    distance = ot.emd2([], [], cost_matrix)

    print(f"Wasserstein distance: {distance}")
    opt_target = configs['opt']['target']
    if opt_target == 'max':
        print(f"Train {target_prop}:{max(train_targets)}")
        print(f"Generated {target_prop}:{max(results_scores)}")
    elif opt_target == 'min':
        print(f"Train {target_prop}:{min(train_targets)}")
        print(f"Generate {target_prop}:{min(results_scores)}")

if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', dest='yaml_path', help='yaml path', required=True, type=str)
    args = parser.parse_args()
    with open(args.yaml_path,'r') as f:
        configs = yaml.safe_load(f)
    main(configs)

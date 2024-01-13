from rdkit import Chem
import numpy as np

import random
import torch
import sascorer
import networkx as nx

from rdkit.Chem import QED

from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(908530)

with open("zinc/train.txt", "r") as f:
    smiles = f.readlines()
smiles = [s.strip("\n\r") for s in smiles]

def get_prop(mol):
    if mol is not None:
        score = (10/sascorer.calculateScore(mol) - 1) / 9
    else:
        score = 10
    return score

num_sample = 10000
samples = []
props = []

while len(props) < num_sample:
    print(len(props))
    smi = random.choice(smiles)

    mol = Chem.MolFromSmiles(smi)
    score = get_prop(mol)

    if smi not in samples:
        samples.append(smi)
        props.append(score)

np.save("opt/gri_psa_train_smiles10k.npy", samples)
np.save("opt/gri_psa_train_props10k.npy", props)
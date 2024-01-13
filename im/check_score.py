from rdkit import Chem
from scorers.qed_scores import score_function
mol = Chem.MolFromSmiles('O=CC(O)c1cccc(NCC2CC=CCC2)c1O')
s = score_function(mol)
print(s)
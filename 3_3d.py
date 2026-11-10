
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. Define SMILES
smiles = "OC1=CC=CC=C1Br"

# 2. Generate molecule
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    raise ValueError("Invalid SMILES")

# 3. Add hydrogens
mol = Chem.AddHs(mol)

# 4. Generate 3D coordinates
params = AllChem.ETKDGv3()
params.randomSeed = 42

status = AllChem.EmbedMolecule(mol, params)
if status != 0:
    raise RuntimeError("3D embedding failed")

# 5. Optimize geometry using MMFF
if AllChem.MMFFHasAllMoleculeParams(mol):
    AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
else:
    AllChem.UFFOptimizeMolecule(mol, maxIters=1000)

# 6. Export PDB
Chem.MolToPDBFile(mol, "cyclohexane_bromo_oh.pdb")

# 7. Export XYZ
conf = mol.GetConformer()

with open("cyclohexane_bromo_oh.xyz", "w") as f:
    f.write(f"{mol.GetNumAtoms()}\n")
    f.write("Cyclohexane with bromine and hydroxyl substituents\n")

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        f.write(
            f"{atom.GetSymbol():<2} "
            f"{pos.x: .6f} "
            f"{pos.y: .6f} "
            f"{pos.z: .6f}\n"
        )

print("Generated PDB and XYZ files successfully.")

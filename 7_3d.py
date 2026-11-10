"""RDKit -> 3D structure files for VMD.  (4R,5S,6S)-5-bromo-4-hydroxy-6-methyloctan-2-one"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdCIPLabeler

SMILES = "C#CC[C@H](C(F)(F)F)[C@H](I)[C@H](Br)CC(=O)C12CC3CC(CC(C3)C1)C2"
NAME = "analog5"

mol = Chem.AddHs(Chem.MolFromSmiles(SMILES))
cids = AllChem.EmbedMultipleConfs(mol, 100, randomSeed=7)
res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=5000)
es = np.array([e for _, e in res])

# Among conformers within 3 kcal/mol of the minimum, take the most extended one (clearer picture).
# Set EXTENDED=False to use the global minimum instead.
EXTENDED = True
def rg(c):
    X = mol.GetConformer(cids[c]).GetPositions()
    return np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean())
best = max([c for c in range(len(es)) if es[c] - es.min() < 3.0], key=rg) if EXTENDED else int(es.argmin())
cid = cids[best]

# verify the 3D geometry still encodes the intended stereochemistry
m3 = Chem.Mol(mol); Chem.AssignStereochemistryFrom3D(m3, confId=cid)
def cip(m):
    m = Chem.Mol(m); rdCIPLabeler.AssignCIPLabels(m)
    return {a.GetIdx(): a.GetProp("_CIPCode") for a in m.GetAtoms() if a.HasProp("_CIPCode")}
orig = cip(mol)
got = cip(m3)
assert {k: got.get(k) for k in orig} == orig, "stereochemistry changed!"
print("CIP:", orig, "| conformer", best, f"| rel. E = {es[best]-es.min():.2f} kcal/mol")

# center on origin (nicer in VMD)
conf = mol.GetConformer(cid)
xyz = conf.GetPositions(); xyz -= xyz.mean(0)
for i, p in enumerate(xyz): conf.SetAtomPosition(i, p.tolist())

for a in mol.GetAtoms():                      # residue info so VMD labels look sane
    a.SetMonomerInfo(Chem.AtomPDBResidueInfo(f"{a.GetSymbol()}{a.GetIdx()+1:<3d}", residueName="LIG", residueNumber=1, isHeteroAtom=True))
Chem.MolToPDBFile(mol, f"{NAME}.pdb", confId=cid)        # includes CONECT records -> bonds in VMD
Chem.MolToXYZFile(mol, f"{NAME}.xyz", confId=cid)
Chem.MolToMolFile(mol, f"{NAME}.mol", confId=cid)


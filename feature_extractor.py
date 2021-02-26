from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops
import numpy
from rdkit import Chem, DataStructs

def fingerprint_features(smile_string, radius=2, size=2048):
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius,
                                                          nBits=size,
                                                          useChirality=True,
                                                          useBondTypes=True,
                                                          useFeatures=False
                                                         )
def convert_to_numpy(fp):
    arr = numpy.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
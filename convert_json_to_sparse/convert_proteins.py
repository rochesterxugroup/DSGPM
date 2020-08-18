import argparse
import json
import numpy as np
import scipy.sparse

from rdkit.Chem import AllChem


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Converts output json files to sparse matrices')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--pdb', type=str, required=True)
    # parser.add_argument('--outputdir',type=str,required=True)

    args = parser.parse_args()
    return args

# currently takes only one file at time.
# TODO: zip pdb folder with database


def sparse_convert(args):
    json_file = args.input
    pdb_file = args.pdb
    chalist = json_file.split('.')
    n = int(chalist[-2])
    with open(json_file, 'r') as myfile:
        obj = json.load(myfile)
    mol = AllChem.MolFromPDBFile(pdb_file)
    m = int(mol.GetNumAtoms())
    map_mat = np.zeros((n, m))
    for nodes in obj['nodes']:
        iid = int(nodes['id'])
        cgid = int(nodes['cg'])
        mass = mol.GetAtomWithIdx(iid).GetMass()
        map_mat[cgid, iid] = mass
    sparse_mat = scipy.sparse.csc_matrix(map_mat)
    outfile2 = json_file.replace(".json", ".npz")
    scipy.sparse.save_npz(outfile2, sparse_mat)
    print(outfile2)
    print('Saved as sparse matrices')


def main():
    args = arg_parse()
    sparse_convert(args)


if __name__ == "__main__":
    main()

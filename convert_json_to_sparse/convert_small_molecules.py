import argparse
import json
import os
import numpy as np
import scipy.sparse

from rdkit import Chem


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Converts output json files to sparse matrices')
    parser.add_argument('--inputdir', type=str, required=True)
    parser.add_argument('--outputdir', type=str, required=True)
    args = parser.parse_args()
    return args


def sparse_convert(args):
    indir = args.inputdir
    outdir = args.outputdir
    filelist = [f for f in os.listdir(
        indir) if os.path.isfile(os.path.join(indir, f))]
    for i in range(0, len(filelist)):
        json_input_name = filelist[i]
        if filelist[i].endswith('.json'):
            chalist = json_input_name.split('.')
            n = int(chalist[-2])
            path = os.path.join(indir, json_input_name)
            with open(path, 'r') as myfile:
                obj = json.load(myfile)
            s = str(obj['smiles'])
            mol = Chem.MolFromSmiles(s)
            m = int(mol.GetNumAtoms())
            map_mat = np.zeros((n, m))
            for nodes in obj['nodes']:
                iid = int(nodes['id'])
                cgid = int(nodes['cg'])
                mass = mol.GetAtomWithIdx(iid).GetMass()
                map_mat[cgid, iid] = mass
            sparse_mat = scipy.sparse.csc_matrix(map_mat)
            outfile2 = json_input_name.replace(".json", ".npz")
            outdir2 = os.path.join(outdir, outfile2)
            scipy.sparse.save_npz(outdir2, sparse_mat)
    print('Saved as sparse matrices')


def main():
    args = arg_parse()
    sparse_convert(args)


if __name__ == "__main__":
    main()

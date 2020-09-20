#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Geemi Wellawatte
#  Email: gwellawa@rochester.edu

import json
import argparse
import os
import re

from rdkit import Chem
from rdkit.Chem import AllChem


def convert_to_json(m, o_fmt, file_dir, sml_string=None):
    '''Takes in a PDB file or a SMILES string and returns one bead mapping
       in JSON format. This is the input to the DGSPM model.

       :m : molecule object generated from PDB file or SMILES using RDKit
       :o_fmt : input format (PDB or SMILES)
       :file_dir : path to the input
       :sml_string : SMILES string
    '''

    edges = []
    for j in range(m.GetNumBonds()):
        begin = m.GetBonds()[j].GetBeginAtomIdx()
        end = m.GetBonds()[j].GetEndAtomIdx()
        bond = m.GetBondWithIdx(j).GetBondTypeAsDouble()
        value = {"source": begin, "target": end, "bondtype": bond}
        edges.append(value)

    # Create one bead mappings
    nodes = []
    cgnodes = []
    for l in range(m.GetNumAtoms()):
        element = m.GetAtomWithIdx(l).GetSymbol()
        val = {"cg": 0, "element": element, "id": l}
        nodes.append(val)
        cgnodes.append(l)

    # Create a nested dictionary to be given in json format
    cg_dict = {"cgnodes": cgnodes, "nodes": nodes, "edges": edges}

    # Writing to json file
    file_path = file_dir.split('/')
    dir_path = '/'.join(file_path[:-1])
    out_dir = os.path.join(dir_path, 'mol_graph')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if o_fmt == 'pdb':
        output_name = file_path[-1].replace(".pdb", ".json")
        out_file = os.path.join(out_dir, output_name)
        with open(out_file, "w") as outfile:
            outfile.write(json.dumps(cg_dict, sort_keys=True, indent=4))

    elif o_fmt == 'smile':
        oname = re.sub('[^A-Za-z0-9]+', '', sml_string) + '.json'
        ofile = os.path.join(out_dir, oname)
        with open(ofile, 'w') as f:
            f.write(json.dumps(cg_dict, sort_keys=True, indent=4))

    print('conversion complete')


def arg_parse():
    '''Takes in the arguments to convert input formats to JSON.
       This scripts reads only one text file containing SMILES.
       Note: Both PDBs and SMILES must be preprocessed to remove Hydrogens in the inputs.

       :--pdb : path to the folder containing PDB files
       :--smiles: path to the folder containing a text file of SMILES
    '''

    parser = argparse.ArgumentParser(
        description='Converts PDB files to JSON inputs')
    parser.add_argument('--pdb', type=str)
    parser.add_argument('--smiles', type=str)
    args = parser.parse_args()

    return args


def main():
    args = arg_parse()

    if args.pdb is None and args.smiles is None:
        print('Speficy input folder')

    elif args.pdb is not None:

        directory = args.pdb
        filelist = [
            i for i in os.listdir(directory) if os.path.isfile(
                os.path.join(
                    directory,
                    i))]

        for f in range(len(filelist)):
            if filelist[f].endswith('.pdb'):
                pdb_path = os.path.join(directory, filelist[f])
                print(pdb_path)
                mol = AllChem.MolFromPDBFile(pdb_path)
                if mol is not None:
                    convert_to_json(mol, 'pdb', pdb_path)
                else:
                    print('Unable to generate molecule. Skipping:', f)

            else:
                print('No PDB files in this folder')

    elif args.smiles is not None:
        path = args.smiles
        s_file = open(path, 'r')
        lines = s_file.readlines()
        for l in range(0, len(lines)):
            sml = str(lines[l])
            print(sml)
            mol = Chem.MolFromSmiles(sml, False)
            if mol is not None:
                convert_to_json(mol, 'smile', path, sml)
            else:
                print('Unable to generate molecule. Skipping:', l)


if __name__ == "__main__":
    main()

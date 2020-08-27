#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Geemi Wellawatte
#  Email: gwellawa@rochester.edu

import rdkit
import json
import numpy as np
import argparse
from rdkit import Chem
import os
from rdkit.Chem import AllChem


def pdb_to_json(pdb_file):
    '''Takes in a pdb file and returns the one bead mapping
       in JSON format. This is the input to the DGSPM model.

       :pdb_file: path to the pdb file
       :output_name: name of the output file

       :return: processed input file in JSON format
    '''

    if pdb_file is None:
        print('Please specify path to the pdb file')
    else:
        m = AllChem.MolFromPDBFile(pdb_file)
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
        file_path = pdb_file.split('/')
        dir_path = '/'.join(file_path[:-1])
        out_dir = os.path.join(dir_path, 'dsgpm_input')

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        output_name = file_path[-1].replace(".pdb", ".json")
        out_file = os.path.join(out_dir, output_name)
        with open(out_file, "w") as outfile:
            outfile.write(json.dumps(cg_dict))

        print('conversion complete')


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Converts PDB files to JSON inputs')
    parser.add_argument('--pdb', type=str, required=True)
    args = parser.parse_args()

    return args


def main():
    args = arg_parse()
    directory = args.pdb
    filelist = [
        f for f in os.listdir(directory) if os.path.isfile(
            os.path.join(
                directory,
                f))]
    for f in range(len(filelist)):
        if filelist[f].endswith('.pdb'):
            pdb_path = os.path.join(directory, filelist[f])
            print(pdb_path)
            pdb_to_json(pdb_path)

        else:
            print('No PDB files in this folder')


if __name__ == "__main__":
    main()

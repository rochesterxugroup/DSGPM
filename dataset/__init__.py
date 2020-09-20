BOND_TYPE_DICT = {1.0: 0, 1.5: 1, 2.0: 2, 3.0: 3, '-': 0, '/': 0, '\\': 0, ':': 1, '=': 2, '#': 3}
MASK_ATOM_INDEX = 0

from .ham import HAM
from .ChEMBL import ChEMBL


def get_num_atoms_by_dataset(dataset_name):
    if dataset_name == 'HAM':
        return len(HAM.ATOMS)
    elif dataset_name == 'ChEMBL':
        return len(ChEMBL.ATOMS)
    else:
        raise NotImplementedError


def get_dataset_class(dataset_name):
    if dataset_name == 'HAM':
        return HAM
    elif dataset_name == 'ChEMBL':
        return ChEMBL
    else:
        raise NotImplementedError

import pandas
import os
import pickle



def pyg_moldataset(d_name):
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.loader import DataLoader

    work_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(work_dir, '../dataset')

    dataset = PygGraphPropPredDataset(d_name, root = data_dir)
    data_name = d_name.replace('-', '_')
    dataset_dir = os.path.join(work_dir, '../dataset', data_name)
    original_data = pandas.read_csv(
        os.path.join(dataset_dir, 'mapping', 'mol.csv.gz'),
        compression = 'gzip'
    )
    smiles = original_data.smiles
    return smiles, dataset


def pyg_molsubdataset(d_name, preprocess_method = 'brics'):
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.loader import DataLoader

    work_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(work_dir, '../dataset')

    dataset = PygGraphPropPredDataset(d_name, root = data_dir)
    data_name = d_name.replace('-', '_')
    dataset_dir = os.path.join(work_dir, '../dataset', data_name)
    original_data = pandas.read_csv(
        os.path.join(dataset_dir, 'mapping', 'mol.csv.gz'),
        compression = 'gzip'
    )
    smiles = original_data.smiles

    pre_name = os.path.join(work_dir, '../../preprocess', data_name)
    pre_file = os.path.join(pre_name, 'substructures.pkl') \
        if preprocess_method == 'brics' else \
        os.path.join(pre_name, 'substructures_recap.pkl')

    if not os.path.exists(pre_file):
        raise IOError('please run preprocess script for dataset')
    with open(pre_file, 'rb') as f:
        substructures = pickle.load(f)

    return smiles, substructures, dataset

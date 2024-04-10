import argparse
import os
from modules.DataLoad import pyg_moldataset
from tqdm import tqdm
import subprocess
import pickle



def get_result_dir():
    work_dir = os.path.abspath(os.path.dirname(__file__))
    result_dir = os.path.join(work_dir, '../preprocess')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocessing for Dataset')
    parser.add_argument(
        '--dataset', default = 'ogbg-molbace', type = str,
        help = 'the dataset to preprocess'
    )
    parser.add_argument(
        '--method', choices = ['brics', 'recap'], default = 'brics',
        help = 'the method to decompose molecules'
    )
    parser.add_argument(
        '--timeout', default = 120, type = int,
        help = 'maximal time to process a single molecule'
    )

    args = parser.parse_args()
    print(args)

    result_dir = get_result_dir()
    data_name = args.dataset.replace('-', '_')
    if not os.path.exists(os.path.join(result_dir, data_name)):
        os.mkdir(os.path.join(result_dir, data_name))
    smiles, dataset = pyg_moldataset(args.dataset)
    file_name = 'substructures.pkl' if args.method == 'brics' \
        else 'substructures_recap.pkl'
    file_name = os.path.join(result_dir, data_name, file_name)
    escapes = []
    with open(file_name, 'w') as f:
        for idx, smile in enumerate(tqdm(smiles)):
            try:
                subprocess.run([
                    'python', 'modules/getSubstructure.py', '--smile',
                    smile, '--method', args.method
                ], check = True, timeout = args.timeout, stdout = f)
            except subprocess.TimeoutExpired:
                escapes.append(idx)
                f.write(f'{smile}\t{str(set([smile]))}\n')

    if len(escapes) > 0:
        print('[INFO] the following molecules are processed unsuccessfully:')
        [print(smiles[x]) for x in escapes]

    substruct_list = []

    with open(file_name) as f:
        for line in f:
            if len(line) <= 1:
                continue
            line = line.strip().split('\t')
            assert len(line) == 2, f'Invalid Line {line}'
            assert type(eval(line[1])) == set, f'Invalid value1 {line[1]}'
            if len(eval(line[1])) == 0:
                print(
                    f'[INFO] empty substruct find for {line[0]},'
                    'consider itself as a substructure'
                )
                substruct_list.append(set(line[0]))
            else:
                substruct_list.append(eval(line[1]))

    with open(file_name, 'wb') as f:
        pickle.dump(substruct_list, f)

import argparse
import os
import json
from tqdm import tqdm
import subprocess



def get_all_smiles(prefix = 'dataset', dataset = 'ic50'):
    with open(os.path.join(prefix, f'lbap_core_{dataset}_assay.json')) as f:
        info = json.load(f)
    parts = ['train', 'iid_test', 'ood_test', 'iid_val', 'ood_val']
    all_smiles = []
    for part in parts:
        all_smiles += [x['smiles'] for x in info['split'][part]]
    with open(os.path.join(prefix, 'all_smiles.json'), 'w') as f:
        json.dump(all_smiles, f, indent = 4)
    return all_smiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocessing for Dataset')
    parser.add_argument(
        '--start', default = 0, type = int,
        help = 'start idx for decomposition'
    )
    parser.add_argument(
        '--num', default = 5000, type = int,
        help = 'the number of molecules for decomposition'
    )
    parser.add_argument(
        '--dataset', choices = ['ic50', 'ec50'], default = 'ic50',
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

    PREFIX = os.path.join('dataset', args.dataset)
    if not os.path.exists(os.path.join(PREFIX, 'all_smiles.json')):
        all_smiles = get_all_smiles(PREFIX, args.dataset)
    else:
        with open(os.path.join(PREFIX, 'all_smiles.json')) as f:
            all_smiles = json.load(f)

    if args.method == 'brics':
        output_dir = os.path.join(PREFIX, 'substructure')
    else:
        output_dir = os.path.join(PREFIX, 'substructure_recap')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f'[INFO] there are {len(all_smiles)} molecules in total')
    assert args.start < len(all_smiles), 'start_idx too large'
    print(f'[INFO] decomposition from {args.start} to {args.start + args.num - 1}')

    out_file_name = f'{args.start}-{args.start + args.num - 1}.txt'
    f = open(os.path.join(output_dir, out_file_name), 'w')
    escapes = []
    for idx, smile in enumerate(tqdm(all_smiles[args.start: args.start + args.num])):
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
        print(escapes)
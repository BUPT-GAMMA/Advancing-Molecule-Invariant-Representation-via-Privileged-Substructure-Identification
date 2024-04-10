import os
import json
from tqdm import tqdm



def process(method):
    result = {}
    print(f'[INFO] Loading substructures from {method}')
    PREFIX = 'substructure' if method == 'brics' else 'substructure_recap'
    for file in os.listdir(PREFIX):
        if not file.endswith(('.json', '.txt')):
            continue
        if file.endswith('.json'):
            with open(os.path.join(PREFIX, file)) as f:
                INFO = json.load(f)
            for k, v in tqdm(INFO.items()):
                assert type(eval(v)) == set, f'Invalid Value {v}'
                result[k] = v
        else:
            with open(os.path.join(PREFIX, file)) as f:
                for line in tqdm(f):
                    if len(line) <= 1:
                        continue
                    line = line.strip().split('\t')
                    assert len(line) == 2, f'Invalid Line {line}'
                    assert type(eval(line[1])) == set, \
                        'Invalid Value {}'.format(line[1])
                    result[line[0]] = line[1]

    with open('all_smiles.json') as f:
        all_smiles = json.load(f)

    new_result, empty_num = {}, 0
    for k, v in result.items():
        v = eval(v)
        if len(v) == 0:
            print(f'[INFO] {k} have no substructures, use it self as substructure')
            new_result[k], empty_num = set([k]), empty_num + 1
        else:
            new_result[k] = v

    print(f'[INFO] there are {empty_num} molcules have no substructures')

    result = {k: str(v) for k, v in new_result.items()}

    for split in ['assay', 'scaffold', 'size']:
        print(f'[INFO] Processing environment split {split}')
        with open(f'lbap_core_ic50_{split}.json') as f:
            INFO = json.load(f)
        for k, v in INFO['split'].items():
            new_mol_list = []
            for mol in tqdm(v):
                mol['substructure'] = result[mol['smiles']]
                new_mol_list.append(mol)
            INFO['split'][k] = new_mol_list
        with open(f'lbap_core_ic50_{split}_{method}.json', 'w') as f:
            json.dump(INFO, f, indent=4)


if __name__ == '__main__':
    if os.path.exists('substructure'):
        process('brics')
    if os.path.exists('substructure_recap'):
        process('recap')
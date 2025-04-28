import os
import subprocess
from ruamel.yaml import YAML

yaml = YAML()

if __name__ == '__main__':

    script_path = 'train_scripts/train_0306.py'
    config_path = 'configs/config_0409_SirajMPRA_3_cell_type.yaml'

    # subprocess.run(
    #     f'OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 {script_path} -c {config_path}',
    #     shell=True)

    # for cell_type in ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']:
    for assay in ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']:
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        
        # config['cell_types'].remove(cell_type)
        # config['cell_types'][0] = cell_type # should not change config['cell_types'] list, because yaml anchor and ref will change

        config['assay'] = assay

        new_config_path = f'{config_path.split(".")[0]}_{assay}.yaml'
        with open(new_config_path, 'w') as f:
            yaml.dump(config, f)

        subprocess.run(
            f'OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 {script_path} -c {new_config_path};',
            # f'python {script_path} --config_path {new_config_path};',
            shell=True)

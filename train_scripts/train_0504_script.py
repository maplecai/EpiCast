import os
import subprocess
from ruamel.yaml import YAML

yaml = YAML()

if __name__ == '__main__':

    script_path = 'train_scripts/train_0504.py'
    config_path_list = [
        'configs/config_0504_Gosai_MPRA_MyResTransformer3c1a_DNase.yaml',
        'configs/config_0504_Gosai_MPRA_MyResTransformer3c1a_H3K4me3.yaml',
        'configs/config_0504_Gosai_MPRA_MyResTransformer3c1a_H3K27ac.yaml',
        'configs/config_0504_Gosai_MPRA_MyResTransformer3c1a_CTCF.yaml',
    ]


    for config_path in config_path_list:
        subprocess.run([
            'python',
            script_path,
            '-c',
            config_path,
        ])


        # subprocess.run([
        #     'torchrun',
        #     '--nproc_per_node=2',
        #     script_path,
        #     '-c', 
        #     config_path,
        # ], env=dict(os.environ, OMP_NUM_THREADS='4'))
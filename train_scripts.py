import subprocess
import yaml

def change_config(config_file, save_dir, key, value):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        config[key] = value

    if save_dir is not None:
        with open(save_dir, 'w') as f:
            yaml.dump(config, f)


if __name__ == '__main__':


    python_path = 'train_DDP_A_valid_B_0414_2.py'
    config_path = 'configs/config_GosaiMPRA_0414_1_manual1.yaml'
    subprocess.run(
        f'export OMP_NUM_THREADS=1;'
        f'export CUDA_VISIBLE_DEVICES=1,2,3;'
        f'torchrun --nproc_per_node=3 {python_path} --config_path {config_path};',
        shell=True)




    python_path = 'train_DDP_A_valid_B_0414_2.py'
    config_path = 'configs/config_GosaiMPRA_0414_1_manual2.yaml'
    subprocess.run(
        f'export OMP_NUM_THREADS=1;'
        f'export CUDA_VISIBLE_DEVICES=1,2,3;'
        f'torchrun --nproc_per_node=3 {python_path} --config_path {config_path};',
        shell=True)




    python_path = 'train_DDP_A_valid_B_0414_2.py'
    config_path = 'configs/config_GosaiMPRA_0414_1_manual012.yaml'
    subprocess.run(
        f'export OMP_NUM_THREADS=1;'
        f'export CUDA_VISIBLE_DEVICES=1,2,3;'
        f'torchrun --nproc_per_node=3 {python_path} --config_path {config_path};',
        shell=True)






    python_path = 'train_DDP_A_valid_B_0414_2.py'
    config_path = 'configs/config_GosaiMPRA_0414_1.yaml'

    key = 'selected_train_datasets_idx'
    value_list = [[3], [1,3], [2,3], [0,1,2,3]]

    for value in value_list:
        value_str = ''.join(str(i) for i in value)
        save_dir = config_path.replace('.yaml', f'_{value_str}.yaml')
        change_config(config_path, save_dir, key, value)

        subprocess.run(
            f'export OMP_NUM_THREADS=1;'
            f'export CUDA_VISIBLE_DEVICES=1,2,3;'
            f'torchrun --nproc_per_node=3 {python_path} --config_path {save_dir};'
            f'rm {save_dir}', 
            shell=True)





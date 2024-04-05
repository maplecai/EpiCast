# torchrun --nproc_per_node=4 train_parallel_A_valid_B2_trainer.py --config_path configs/config_GosaiMPRA_onehot_CNN_joint_BassetN_small.yaml
torchrun --nproc_per_node=4 train_DDP_A_valid_B_trainer.py --config_path configs/config_Xpresso.yaml

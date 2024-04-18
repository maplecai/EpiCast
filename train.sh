#!/bin/bash

# torchrun --nproc_per_node=4 train_DDP_A_valid_B_trainer.py --config_path configs/config_Xpresso.yaml
# torchrun --nproc_per_node=4 train_DDP_AC_valid_BD_trainer.py --config_path configs/config_Xpresso_GosaiMPRA.yaml
# torchrun --nproc_per_node=4 train_DDP_AC_valid_BD_multitask.py --config_path configs/config_Xpresso_GosaiMPRA_multitask.yaml

# torchrun --nproc_per_node=4 train_DDP_A_valid_B_trainer_multicelltype.py --config_path configs/config_GosaiMPRA_multicelltype.yaml
# # torchrun --nproc_per_node=4 train_DDP_A_valid_B_trainer_onecelltype.py --config_path configs/config_GosaiMPRA_onecelltype.yaml

torchrun --nproc_per_node=4 train_DDP_A_valid_B_trainer_multimulti_loader.py --config_path configs/config_GosaiMPRA_multimulti_loader_subset_att.yaml
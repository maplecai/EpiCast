cd ..
# python predict_epi_features/0_predict_Sei_feature.py --data_path data/Gosai_MPRA/Gosai_MPRA_designed.csv --output_path predict_epi_features/outputs/Gosai_MPRA_designed_Sei_pred_0528.h5 -m Sei -d cuda:0
python predict_epi_features/0_predict_epi_feature.py --data_path 'data/Agarwal_MPRA/Agarwal_MPRA_joint_56k.csv' --output_path predict_epi_features/outputs/Agarwal_Enformer_pred.h5 -m Enformer -d cuda:0

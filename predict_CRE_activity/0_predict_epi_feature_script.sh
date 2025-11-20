cd ..
# python predict_CRE_activity/0_predict_Sei_feature.py --input_path data/Gosai_MPRA/Gosai_MPRA_designed.csv --output_path predict_CRE_activity/outputs/Gosai_MPRA_designed_Sei_pred_0528.h5 -m Sei -d cuda:0
# python predict_CRE_activity/0_predict_epi_feature.py --input_path 'data/Agarwal_MPRA/Agarwal_MPRA_joint_56k.csv' --output_path predict_CRE_activity/outputs/Agarwal_Enformer_pred.h5 -m Enformer -d cuda:0
# python predict_CRE_activity/0_predict_epi_feature.py --input_path 'data/cCRE/cCRE_chr3.csv' --output_path predict_CRE_activity/outputs/cCRE_chr3_pred.h5 -m Sei -d cuda:1
python predict_CRE_activity/0_predict_epi_feature.py --input_path 'data/Zhang_MPRA/Zhang_MPRA.csv' --output_path predict_CRE_activity/outputs/Zhang_MPRA_Sei_pred.h5 -m Sei -d cuda:0
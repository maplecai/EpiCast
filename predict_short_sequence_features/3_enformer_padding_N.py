import sys
sys.path.append("..")
from MPRA_predict.utils import *
from MPRA_predict.datasets import *

from torch.utils.data import DataLoader
from enformer_pytorch import from_pretrained


for cropped_length in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608]:
    output_path = f'output/Enformer_pred_crop_{cropped_length}_pad_196608_N.npy'
    model_path = f'../pretrained_models/enformer_pytorch'
    data_path = f'data/enformer_sequences_test.csv'

    if not os.path.exists(output_path):
        model = from_pretrained(model_path, target_length=2)
        dataset = SeqLabelDataset(
            data_path=data_path,
            data_type='seq',
            input_column='seq', 
            crop=True, ###
            crop_method='center',
            cropped_length=cropped_length,
            padding=True, ###
            padding_method='N',
            padded_length=196608,
            N_fill_value=0.25)
        
        test_data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
        pred = get_pred(model, test_data_loader)
        np.save(output_path, pred)

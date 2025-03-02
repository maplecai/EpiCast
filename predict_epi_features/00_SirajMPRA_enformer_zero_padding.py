import sys
sys.path.append("..")
from MPRA_predict.utils import *
from MPRA_predict.datasets import *

from torch.utils.data import DataLoader
from MPRA_predict.models.enformer_pytorch import from_pretrained


# only get center pos pred
def get_pred(model, test_data_loader, device='cuda'):
    model = model.to(device)
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_data_loader):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            elif isinstance(batch, dict):
                x = batch['seq']
            x = x.to(device)
            output = model(x)
            output = output['human'][:, 447:449]
            y_pred.append(output.detach().cpu().numpy())
            del batch, x, output  # 清理内存
            torch.cuda.empty_cache() # 清理显存
    y_pred = np.concatenate(y_pred, axis=0)
    return y_pred



if __name__ == '__main__':

    set_seed(0)
    model_path = f'../pretrained_models/enformer_weights'
    data_path = f'../data/SirajMPRA/SirajMPRA_563k.csv'
    output_path = f'outputs/SirajMPRA_Enformer_zero_padding.npy'
    device = f'cuda:1'

    if os.path.exists(output_path):
        print(f'warning, already exists {output_path}')
    print(f'predicting {output_path}')

    model = from_pretrained(model_path)

    dataset = SeqDataset(
        data_path=data_path,
        input_column='seq', 
        crop=False,
        padding=True,
        padding_method='N',
        padded_length=196608,
        N_fill_value=0)
    
    test_data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1)
    pred = get_pred(model, test_data_loader, device)
    np.save(output_path, pred)

    # num_splits = 10    # num_splits是你要分割的部分数
    # split_size = len(dataset) // num_splits
    # for i in range(num_splits):
    #     start_idx = i * split_size
    #     end_idx = (i + 1) * split_size if i != num_splits - 1 else len(dataset)

    #     subset = Subset(dataset, range(start_idx, end_idx))
    #     subloader = DataLoader(subset, batch_size=6, shuffle=False, num_workers=0)
    #     y_pred = get_pred(model, subloader, device)
    #     np.save(f'{output_path}_{i}.npy', np.array(y_pred))

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))
from MPRA_predict import models, datasets, metrics, utils
from MPRA_predict.utils import *

# df = pd.read_csv(ROOT_DIR/'data/Sei/Sei_tracks_info.csv')
# df_pivot = df.pivot_table(
#     values="index", 
#     index="cell_type", 
#     columns="assay", 
#     aggfunc=list,
# )
# df_pivot = df_pivot.map(lambda x: x if isinstance(x, list) else [])
# # df_pivot.map(len)

# assays = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']
# mask = df_pivot[assays].map(lambda x: len(x) > 0).all(axis=1)
# df_track = df_pivot.loc[mask, assays]
# df_track.to_json(ROOT_DIR/'data/Sei/Sei_61_cell_types_tracks.json')

def generate_VEF_from_feature(pred_array, df_track):

    cell_types = df_track.index.tolist()
    assays = df_track.columns.tolist()

    VEF_dict = {}
    for i, cell_type in enumerate(cell_types):
        for j, assay in enumerate(assays):
            indice = df_track.loc[cell_type, assay]
            if isinstance(indice, list) and len(indice) > 0:
                pred = logit(pred_array[:, indice]).mean(1)
                VEF_dict[f'{cell_type}_{assay}'] = pred
            else:
                VEF_dict[f'{cell_type}_{assay}'] = np.NaN
    VEF_df = pd.DataFrame(VEF_dict)
    print(f'{pred_array.shape = }')
    print(f'{len(cell_types) = }, {len(assays) = }')
    print(f'{VEF_df.shape = }')
    return VEF_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True, type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    suffix = input_path.suffix.lower()
    if suffix == ".h5":
        pred_array = load_h5(input_path)
    elif suffix == ".pkl":
        pred_array = load_pickle(input_path)
    elif suffix == ".npy":
        pred_array = np.load(input_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    df_track = pd.read_json(ROOT_DIR / 'data/Sei/Sei_61_cell_types_tracks.json')
    VEF_df = generate_VEF_from_feature(pred_array, df_track)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    VEF_df.to_csv(output_path, sep='\t', index=False)


if __name__ == "__main__":
    main()

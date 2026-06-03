# import h5py
# import epicast
# from epicast import utils, metrics

# import h5py

# src_files = ["alphagenome/gosai_ag_pred_256bp_concat.h5", "alphagenome/last.h5"]
# out_file = "alphagenome/gosai_ag_pred_256bp_concat_2.h5"

# # 先用第一个文件确定 dataset 结构
# with h5py.File(src_files[0], "r") as f:
#     keys = list(f.keys())
#     dtypes = {k: f[k].dtype for k in keys}
#     shapes = {k: f[k].shape[1:] for k in keys}  # 去掉第一维 batch

# # 统计总长度
# total_n = 0
# for fp in src_files:
#     with h5py.File(fp, "r") as f:
#         total_n += int(f.attrs["num_written"])

# # 创建输出文件
# with h5py.File(out_file, "w") as fout:
#     out = {
#         k: fout.create_dataset(
#             k,
#             shape=(total_n, *shapes[k]),
#             dtype=dtypes[k],
#             compression="gzip",
#             shuffle=True,
#         )
#         for k in keys
#     }

#     # 逐个文件拷贝
#     start = 0
#     for fp in src_files:
#         with h5py.File(fp, "r") as f:
#             n = int(f.attrs["num_written"])
#             for k in keys:
#                 out[k][start:start+n] = f[k][:n]
#         start += n

#     fout.attrs["num_written"] = start

# print("done")

import h5py

src_files = [
    "alphagenome/gosai_ag_pred_256bp_concat.h5",
    "alphagenome/last.h5",
]
out_file = "alphagenome/gosai_ag_pred_256bp_concat_2.h5"

with h5py.File(src_files[0], "r") as f:
    keys = list(f.keys())
    dtypes = {k: f[k].dtype for k in keys}
    shapes = {k: f[k].shape[1:] for k in keys}

total_n = 0
for fp in src_files:
    with h5py.File(fp, "r") as f:
        total_n += int(f.attrs["num_written"])

with h5py.File(out_file, "w") as fout:
    out = {
        k: fout.create_dataset(
            k,
            shape=(total_n, *shapes[k]),
            maxshape=(None, *shapes[k]),   # 以后可继续追加
            dtype=dtypes[k],
            compression="gzip",
            shuffle=True,
            chunks=True,
        )
        for k in keys
    }

    start = 0
    for fp in src_files:
        with h5py.File(fp, "r") as f:
            n = int(f.attrs["num_written"])
            for k in keys:
                out[k][start:start+n] = f[k][:n]
        start += n

    fout.attrs["num_written"] = start

print("done")
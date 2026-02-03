import subprocess


subprocess.run(
    ['python', 'scripts/test.py', 
    '-s', 'saved/0123_Gosai_ConvTransFeature_AG_VEF.yaml/0123_021357',
    '-o', 'Gosai_pred.npy',
    '-de', 'cuda:0'],
    cwd='..'
)

# subprocess.run(
#     ['python', 'scripts/test.py', 
#     '-s', 'saved/0123_Gosai_ConvTrans_AG_VEF.yaml/0127_021138',
#     '-o', 'Gosai_pred.npy',
#     '-de', 'cuda:1'],
#     cwd='..'
# )
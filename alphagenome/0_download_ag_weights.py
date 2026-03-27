from huggingface_hub import hf_hub_download
hf_hub_download('gtca/alphagenome_pytorch', 'model_all_folds.safetensors', local_dir='.')
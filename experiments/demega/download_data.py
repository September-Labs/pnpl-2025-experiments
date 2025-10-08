from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="wordcab/libribrain-meg-preprocessed",
    repo_type="dataset",                 
    allow_patterns=["data/grouped_100**"],    
    local_dir="data" 
)
print(local_path)
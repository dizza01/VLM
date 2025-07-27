import os
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from tqdm import tqdm

# Try both possible config locations
config_paths = [
    os.path.join(os.path.dirname(__file__), '../hf_config.yml'),
    os.path.join(os.path.dirname(__file__), '../vlm3d_challenge/hf_config.yml')
]
config = None
for path in config_paths:
    if os.path.exists(path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        break
if config is None:
    raise FileNotFoundError("hf_config.yml not found in VLM or vlm3d_challenge folders.")

token = config['huggingface_token']
repo_id = "ibrahimhamamci/CT-RATE"
directory_name = "dataset/train/"

# Load CT-RATE dataset splits from Hugging Face
reports_ds = load_dataset("ibrahimhamamci/CT-RATE", "reports")
metadata_ds = load_dataset("ibrahimhamamci/CT-RATE", "metadata")
labels_ds = load_dataset("ibrahimhamamci/CT-RATE", "labels")

# Save VolumeName list from train split to CSV
train_vol_names = pd.DataFrame({'VolumeName': reports_ds['train']['VolumeName']})
train_vol_names.to_csv("train_labels.csv", index=False)

# Read the CSV with VolumeName column
data = pd.read_csv("train_labels.csv")

# Only download a small sample for testing
sample_size = 5
sample_vol_names = data["VolumeName"].head(sample_size)

for name in tqdm(sample_vol_names, desc=f"Downloading {sample_size} CT volumes"):
    folder1 = name.split("_")[0]
    folder2 = name.split("_")[1]
    folder = folder1 + "_" + folder2
    folder3 = name.split("_")[2]
    subfolder = folder + "_" + folder3
    subfolder = directory_name + folder + "/" + subfolder
    
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        subfolder=subfolder,
        filename=name,
        local_dir="data_volumes"
    )

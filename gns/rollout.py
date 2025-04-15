import os
import hydra
from omegaconf import DictConfig
import torch
from train import predict_multiple_files

@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Example list of .npz files to process
    # Replace these paths with your actual .npz file paths
    npz_files = [
        "/mnt/c/Users/baage/Documents/tmp/random_field_case1/trajectory_mpm_full_sim_0.npz",
        "/mnt/c/Users/baage/Documents/tmp/random_field_case1/trajectory_mpm_full_sim_10.npz",
        "/mnt/c/Users/baage/Documents/tmp/random_field_case1/trajectory_mpm_full_sim_20.npz"
    ]
    
    # Make sure the files exist
    for file_path in npz_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Call predict_multiple_files
    # You can specify save_format as either "pkl" or "npz"
    predict_multiple_files(device, cfg, npz_files, save_format="pkl")

if __name__ == "__main__":
    main() 
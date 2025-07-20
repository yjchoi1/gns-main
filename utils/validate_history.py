"""
Post-hoc Validation History Evaluation for GNS

This script evaluates multiple model checkpoints on validation data to compute
validation loss history that was not recorded during training.

Key features:
- Evaluates multiple model checkpoints sequentially
- Uses multiple validation samples (instead of just one) for more stable estimates
- Computes mean and standard deviation of validation loss
- Saves detailed results to JSON file
- Supports both .npz and .h5 data formats

Usage:
1. Modify the input parameters at the top of the script
2. Run: python validate_history.py
"""

import json
import os
import sys
import random
import re
from typing import List

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path to import from gns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import particle_data_loader as pdl

# ===== INPUT PARAMETERS - MODIFY THESE =====
# List of model checkpoint files to evaluate
# Example: If your models are saved as model-1000.pt, model-2000.pt, etc.
MODEL_CHECKPOINTS = [
    "/path/to/models/model-1000.pt",
    "/path/to/models/model-2000.pt", 
    "/path/to/models/model-3000.pt",
    "/path/to/models/model-4000.pt",
    "/path/to/models/model-5000.pt",
]

# Path to validation dataset (supports both .npz and .h5 formats)
VALIDATION_DATA_PATH = "/path/to/data/valid.npz"  # or .h5

# Path to data directory (where metadata.json is located)
DATA_PATH = "/path/to/data/"
METADATA_FILE = "metadata.json"

# Number of validation samples to use for each checkpoint
# Increase this for more stable validation estimates, decrease for faster evaluation
NUM_VALIDATION_SAMPLES = 10

# Configuration parameters (adjust these to match your training configuration)
CONFIG = {
    "data": {
        "input_sequence_length": 6,      # Must match your training config
        "noise_std": 6.7e-4,            # Must match your training config
        "kinematic_particle_id": 3,      # Must match your training config
        "num_particle_types": 9,         # Must match your training config
        "batch_size": 2                  # Can be adjusted for efficiency
    }
}

# Device to use for evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output file to save results
OUTPUT_FILE = "validation_history.json"
# ============================================


def extract_step_from_model_path(model_path: str) -> int:
    """Extract step number from model filename."""
    filename = os.path.basename(model_path)
    match = re.search(r'model-(\d+)\.pt', filename)
    if match:
        return int(match.group(1))
    else:
        return 0  # Default if step cannot be extracted


def get_data_file_path(data_path: str, split: str) -> str:
    """Get the appropriate data file path, checking for both .h5 and .npz formats."""
    h5_path = f"{data_path}{split}.h5"
    npz_path = f"{data_path}{split}.npz"
    
    if os.path.exists(h5_path):
        return h5_path
    elif os.path.exists(npz_path):
        return npz_path
    else:
        raise FileNotFoundError(f"Neither {h5_path} nor {npz_path} exists")


def _get_simulator(
    metadata: dict,
    num_particle_types: int,
    acc_noise_std: float,
    vel_noise_std: float,
    device: torch.device,
) -> learned_simulator.LearnedSimulator:
    """Instantiates the simulator."""
    
    # Normalization stats
    normalization_stats = {
        "acceleration": {
            "mean": torch.FloatTensor(metadata["acc_mean"]).to(device),
            "std": torch.sqrt(
                torch.FloatTensor(metadata["acc_std"]) ** 2 + acc_noise_std**2
            ).to(device),
        },
        "velocity": {
            "mean": torch.FloatTensor(metadata["vel_mean"]).to(device),
            "std": torch.sqrt(
                torch.FloatTensor(metadata["vel_std"]) ** 2 + vel_noise_std**2
            ).to(device),
        },
    }

    # Get necessary parameters for loading simulator
    if "nnode_in" in metadata and "nedge_in" in metadata:
        nnode_in = metadata["nnode_in"]
        nedge_in = metadata["nedge_in"]
    else:
        nnode_in = 37 if metadata["dim"] == 3 else 30
        nedge_in = metadata["dim"] + 1

    # Init simulator
    simulator = learned_simulator.LearnedSimulator(
        particle_dimensions=metadata["dim"],
        nnode_in=nnode_in,
        nedge_in=nedge_in,
        latent_dim=128,
        nmessage_passing_steps=10,
        nmlp_layers=2,
        mlp_hidden_dim=128,
        connectivity_radius=metadata["default_connectivity_radius"],
        boundaries=np.array(metadata["bounds"]),
        normalization_stats=normalization_stats,
        nparticle_types=num_particle_types,
        particle_type_embedding_size=16,
        boundary_clamp_limit=(
            metadata["boundary_augment"] if "boundary_augment" in metadata else 1.0
        ),
        device=device,
    )

    return simulator


def prepare_data(example, device):
    """Prepare data for validation."""
    position = example[0][0].to(device)
    particle_type = example[0][1].to(device)

    if len(example[0]) == 4:  # if data loader includes material_property
        material_property = example[0][2].to(device)
        n_particles_per_example = example[0][3].to(device)
    elif len(example[0]) == 3:
        material_property = None
        n_particles_per_example = example[0][2].to(device)
    else:
        raise ValueError("Unexpected number of elements in the data loader")

    labels = example[1].to(device)

    return position, particle_type, material_property, n_particles_per_example, labels


def acceleration_loss(pred_acc, target_acc, non_kinematic_mask):
    """Compute the loss between predicted and target accelerations."""
    loss = (pred_acc - target_acc) ** 2
    loss = loss.sum(dim=-1)
    num_non_kinematic = non_kinematic_mask.sum()
    loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss))
    loss = loss.sum() / num_non_kinematic
    return loss


def validate_single_example(simulator, example, n_features, config, device):
    """Validate a single example and return the loss."""
    (
        position,
        particle_type,
        material_property,
        n_particles_per_example,
        labels,
    ) = prepare_data(example, device)

    # Sample the noise to add to the inputs
    sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
        position, noise_std_last_step=config["data"]["noise_std"]
    ).to(device)
    non_kinematic_mask = (
        (particle_type != config["data"]["kinematic_particle_id"]).clone().detach().to(device)
    )
    sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

    # Get the predictions and target accelerations
    with torch.no_grad():
        pred_acc, target_acc = simulator.predict_accelerations(
            next_positions=labels.to(device),
            position_sequence_noise=sampled_noise.to(device),
            position_sequence=position.to(device),
            nparticles_per_example=n_particles_per_example.to(device),
            particle_types=particle_type.to(device),
            material_property=(
                material_property.to(device) if n_features == 3 else None
            ),
        )

    # Compute loss
    loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)
    return loss


def evaluate_model_checkpoint(model_path, metadata, config, validation_samples, n_features, device):
    """Evaluate a single model checkpoint on validation samples."""
    step = extract_step_from_model_path(model_path)
    print(f"Evaluating model: {os.path.basename(model_path)} (step {step})")
    
    # Create simulator
    simulator = _get_simulator(
        metadata,
        config["data"]["num_particle_types"],
        config["data"]["noise_std"],
        config["data"]["noise_std"],
        device,
    )
    
    # Load model
    simulator.load(model_path)
    simulator.to(device)
    simulator.eval()
    
    # Evaluate on validation samples
    losses = []
    for i, example in enumerate(validation_samples):
        loss = validate_single_example(simulator, example, n_features, config, device)
        losses.append(loss.item())
    
    # Compute mean and std
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    
    print(f"  Step {step}: Mean validation loss: {mean_loss:.6f}")
    print(f"  Step {step}: Std validation loss: {std_loss:.6f}")
    
    return {
        "step": step,
        "model_path": model_path,
        "mean_loss": float(mean_loss),
        "std_loss": float(std_loss),
        "individual_losses": losses,
        "num_samples": len(losses)
    }


def main():
    """Main function to evaluate validation history."""
    print("Starting validation history evaluation...")
    print(f"Device: {DEVICE}")
    print(f"Number of model checkpoints: {len(MODEL_CHECKPOINTS)}")
    print(f"Number of validation samples per checkpoint: {NUM_VALIDATION_SAMPLES}")
    
    # Read metadata
    metadata = reading_utils.read_metadata(DATA_PATH, "train", METADATA_FILE)
    
    # Load validation dataset
    print(f"Loading validation dataset: {VALIDATION_DATA_PATH}")
    valid_dl = pdl.get_data_loader(
        file_path=VALIDATION_DATA_PATH,
        mode="sample",
        input_sequence_length=CONFIG["data"]["input_sequence_length"],
        batch_size=CONFIG["data"]["batch_size"],
        use_dist=False,
    )
    
    # Determine number of features
    valid_dataset = pdl.ParticleDataset(VALIDATION_DATA_PATH)
    n_features = valid_dataset.get_num_features()
    print(f"Number of features in validation dataset: {n_features}")
    
    # Sample validation examples
    print(f"Sampling {NUM_VALIDATION_SAMPLES} validation examples...")
    all_validation_examples = list(valid_dl)
    
    if len(all_validation_examples) < NUM_VALIDATION_SAMPLES:
        print(f"Warning: Only {len(all_validation_examples)} examples available, using all of them")
        validation_samples = all_validation_examples
    else:
        validation_samples = random.sample(all_validation_examples, NUM_VALIDATION_SAMPLES)
    
    print(f"Using {len(validation_samples)} validation samples")
    
    # Sort model checkpoints by step number for better organization
    sorted_checkpoints = sorted(MODEL_CHECKPOINTS, key=extract_step_from_model_path)
    
    # Evaluate each model checkpoint
    results = []
    for model_path in tqdm(sorted_checkpoints, desc="Evaluating checkpoints"):
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
            continue
            
        try:
            result = evaluate_model_checkpoint(
                model_path, metadata, CONFIG, validation_samples, n_features, DEVICE
            )
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {model_path}: {e}")
            continue
    
    # Save results
    output_data = {
        "validation_history": results,
        "configuration": {
            "num_validation_samples": len(validation_samples),
            "validation_data_path": VALIDATION_DATA_PATH,
            "data_path": DATA_PATH,
            "metadata_file": METADATA_FILE,
            "device": str(DEVICE),
            "config": CONFIG
        },
        "metadata": metadata
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\nValidation history saved to: {OUTPUT_FILE}")
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION HISTORY SUMMARY")
    print("="*70)
    print(f"{'Step':<8} {'Model':<20} {'Mean Loss':<12} {'Std Loss':<12}")
    print("-"*70)
    
    # Sort results by step for display
    results_sorted = sorted(results, key=lambda x: x['step'])
    for result in results_sorted:
        model_name = os.path.basename(result["model_path"])
        print(f"{result['step']:<8} {model_name:<20} {result['mean_loss']:<12.6f} {result['std_loss']:<12.6f}")
    print("="*70)


if __name__ == "__main__":
    main() 
# generate_submission.py on holdout dataset

import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pnpl.datasets import LibriBrainCompetitionHoldout
import sys
import importlib
import subprocess
import os

# Add parent directory to path for meg_classifier imports
sys.path.append(str(Path(__file__).parent.parent))
from meg_classifier.models import SingleStageMEGClassifier


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_from_checkpoint(checkpoint_path, config):
    """Load model from checkpoint using config for dynamic model loading.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        
    Returns:
        Loaded model instance
    """
    model_config = config['model']
    
    # Import the specified module
    module_path = model_config['module']
    class_name = model_config['class']
    
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not import {class_name} from {module_path}: {e}")
    
    print(f"Loading {class_name} from checkpoint: {checkpoint_path}")
    
    # Load from checkpoint
    model = model_class.load_from_checkpoint(checkpoint_path)

    # CRITICAL: Set up for inference
    if hasattr(model, 'set_inference_mode'):
        model.set_inference_mode()
        model.use_inference_forward = True
    else:
        model.eval()
    
    return model

def generate_predictions(model, dataset, batch_size=32, num_workers=4, device='cpu', use_tta=True):
    """Generate predictions with optional TTA."""
    model = model.to(device)
    model.eval()
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    predictions = []
    
    print(f"Generating predictions for {len(dataset)} segments...")
    print(f"Using TTA: {use_tta}")
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Processing batches"):
            batch_data = batch_data.to(device)
            
            if use_tta and hasattr(model, 'predict_with_tta'):
                probs = model.predict_with_tta(batch_data)
            else:
                print("Using original model")
                logits = model(batch_data)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                probs = torch.softmax(logits, dim=1)
            
            # Store predictions
            probs_cpu = probs.cpu()
            for i in range(probs_cpu.shape[0]):
                predictions.append(probs_cpu[i])
    
    return predictions

# def generate_predictions(model, dataset, batch_size=32, num_workers=4, device='cpu'):
#     """Generate predictions for competition holdout data."""
#     model = model.to(device)
#     model.eval()  # Ensure eval mode
    
#     # Debug first batch
#     first_batch = torch.stack([dataset[i] for i in range(min(4, len(dataset)))]).to(device)
#     print("\nTesting on first batch:")
#     with torch.no_grad():
#         test_output = model.forward_inference(first_batch, debug=True) if hasattr(model, 'forward_inference') else model(first_batch)
#         #print(f"Test output shape: {test_output.shape}")
    
#     # Create data loader
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers
#     )
    
#     predictions = []
    
#     print(f"Generating predictions for {len(dataset)} segments...")
#     print(f"Using batch size: {batch_size}, device: {device}")
    
#     with torch.no_grad():
#         for batch_data in tqdm(dataloader, desc="Processing batches"):
#             # batch_data shape: (batch_size, 306, 125)
#             batch_data = batch_data.to(device)
            
#             # Forward pass
#             logits = model(batch_data)
#             # Some models return multiple outputs; extract primary logits
#             if isinstance(logits, (tuple, list)):
#                 logits = logits[0]
#             elif isinstance(logits, dict):
#                 logits = logits.get('phoneme', next(iter(logits.values())))

#             # Ensure 2D tensor
#             if logits.dim() == 1:
#                 logits = logits.unsqueeze(0)

#             # Expand to full 39-class space if needed
#             if logits.size(1) != 39:
#                 batch_size_actual = logits.size(0)
#                 full_logits = torch.full((batch_size_actual, 39), float('-inf'), device=logits.device)
#                 # Use model-provided mapping if available
#                 target_indices = getattr(model, 'struggling_indices', list(range(logits.size(1))))
#                 # Safeguard length mismatch
#                 target_indices = list(target_indices)[:logits.size(1)]
#                 index_tensor = torch.as_tensor(target_indices, device=logits.device, dtype=torch.long)
#                 full_logits.scatter_(1, index_tensor.unsqueeze(0).expand(batch_size_actual, -1), logits)
#                 logits = full_logits

#             probs = torch.softmax(logits, dim=1)  # Convert to probabilities
            
#             # Move back to CPU and store individual predictions
#             probs_cpu = probs.cpu()
#             for i in range(probs_cpu.shape[0]):
#                 predictions.append(probs_cpu[i])  # Shape: (39,)
    
#     print(f"Generated {len(predictions)} predictions")
#     return predictions


def submit_to_evalai(submission_file_path, method_name=None):
    """Submit the generated file to EvalAI competition.
    
    Args:
        submission_file_path: Path to the submission CSV file
        method_name: Optional method name for the submission
        
    Returns:
        True if submission was successful, False otherwise
    """
    # Extract method name from filename if not provided
    if method_name is None:
        filename = Path(submission_file_path).stem  # Get filename without extension
        # Remove "_submission" suffix if present
        if filename.endswith("_submission"):
            method_name = filename[:-11]
        else:
            method_name = filename
    
    print(f"\n🚀 Submitting to EvalAI competition...")
    print(f"📝 Method name: {method_name}")
    print(f"📁 Submission file: {submission_file_path}")
    
    # Check file size
    file_size_mb = os.path.getsize(submission_file_path) / (1024 * 1024)
    print(f"📊 File size: {file_size_mb:.2f} MB")
    
    # Build the evalai command
    # Use --large flag only if file is > 400MB
    cmd = [
        "evalai",
        "challenge", "2504",
        "phase", "4972",
        "submit",
        "--file", str(submission_file_path),
        "--private"  # Make submission private
    ]
    
    if file_size_mb > 400:
        cmd.insert(-1, "--large")  # Add --large before --private
        print("📦 Using --large flag for large file upload")
    
    try:
        # Execute the command with real-time output
        print(f"\nExecuting: {' '.join(cmd)}")
        print("⏳ Submitting to EvalAI...")
        print("-" * 60)
        
        # Use Popen for real-time output and to handle prompts
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Send "N" responses for any prompts
        # EvalAI asks for submission details and metadata
        stdin_input = "N\nN\n"
        
        # Communicate with the process
        stdout, _ = process.communicate(input=stdin_input)
        
        # Print the output
        if stdout:
            print(stdout)
        
        print("-" * 60)
        
        # Check if submission was successful
        if "successfully submitted" in stdout.lower():
            print("\n✅ Submission successful!")
            
            # Extract submission ID if present
            import re
            match = re.search(r'ID (\d+)', stdout)
            if match:
                submission_id = match.group(1)
                print(f"📋 Submission ID: {submission_id}")
                print(f"💡 Check status with: evalai submission {submission_id}")
            
            return True
        elif process.returncode == 0:
            print("\n✅ Command completed (check output above)")
            return True
        else:
            print(f"\n❌ Submission failed with error code {process.returncode}")
            return False
            
    except FileNotFoundError:
        print("\n❌ Error: evalai CLI not found!")
        print("Please install it with: pip install evalai")
        print("Then set your token with: evalai set_token <your_token>")
        return False
    except KeyboardInterrupt:
        print("\n\nWARNING: Submission interrupted by user")
        print("The upload may still be in progress on EvalAI's servers")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def main(args):
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to meta directory
        config_path = Path(__file__).parent / "configs" / args.config
    
    if not config_path.exists():
        raise ValueError(f"Config file not found: {args.config}")
    
    config = load_config(config_path)
    
    # Get configurations
    submission_config = config['submission']
    data_config = config['data']
    
    # Load model from checkpoint
    model = load_model_from_checkpoint(args.checkpoint, config)
    model.eval()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using model: {config['model']['class']} from {config['model']['module']}")
    
    # Load competition holdout dataset
    print("Loading phoneme holdout dataset...")
    holdout_dataset = LibriBrainCompetitionHoldout(
        data_path=data_config['data_path'],
        task=submission_config['holdout_task'],
        tmin=data_config['tmin'],
        tmax=data_config['tmax'],
        standardize=False
    )
    
    print(f"Dataset loaded: {len(holdout_dataset)} segments")
    print(f"Each segment shape: {holdout_dataset[0].shape}")  # Should be (306, 125)
    
    # Generate predictions
    predictions = generate_predictions(
        model=model,
        dataset=holdout_dataset,
        batch_size=submission_config['batch_size'],
        num_workers=submission_config['num_workers'],
        device=device
    )
    
    # Create submission file
    output_path = Path(submission_config['output_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    holdout_dataset.generate_submission_in_csv(predictions, str(output_path))
    
    print(f"\n✅ Submission file created: {output_path}")
    print(f"📊 Contains {len(predictions)} predictions")
    
    # Validate submission format
    print("\nValidating submission format...")
    with open(output_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)} (1 header + {len(lines)-1} predictions)")
    print(f"Header: {lines[0].strip()}")
    print(f"First prediction: {lines[1].strip()[:100]}...")
    
    # Check probabilities sum to 1
    first_pred = lines[1].strip().split(',')
    prob_sum = sum(float(p) for p in first_pred[1:])  # Skip segment_idx
    print(f"First prediction probability sum: {prob_sum:.6f}")
    
    if abs(prob_sum - 1.0) < 1e-5:
        print("✅ Probabilities sum to 1.0 (valid submission)")
    else:
        print("WARNING: Probabilities do not sum to 1.0!")
    
    # Submit to EvalAI if requested
    if args.instant_send:
        print("\n" + "="*60)
        print("INSTANT SUBMISSION TO EVALAI")
        print("="*60)
        
        # Check if evalai token is set
        try:
            token_check = subprocess.run(
                ["evalai", "get_token"],
                capture_output=True,
                text=True,
                check=False
            )
            if token_check.returncode != 0 or "Token not found" in token_check.stdout:
                print("Warning: EvalAI token might not be set.")
                print("Set it with: evalai set_token <your_token>")
                response = input("\nContinue with submission anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Submission cancelled.")
                    return
        except:
            pass
        
        # Submit with optional method name override
        method_name = args.method_name if args.method_name else None
        success = submit_to_evalai(output_path, method_name)
        
        if success:
            print("\n🎯 Submission complete! Check your results on EvalAI.")
        else:
            print("\nWARNING: Submission failed. Please submit manually:")
            print(f"evalai challenge 2504 phase 4972 submit --file {output_path} --large --private")
    else:
        print("\n📌 To submit to EvalAI, run:")
        print(f"evalai challenge 2504 phase 4972 submit --file {output_path} --large --private")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate competition submission for phoneme classification")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--instant-send",
        action="store_true",
        help="Instantly submit to EvalAI after generating predictions"
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default=None,
        help="Optional method name for the submission (defaults to output filename without '_submission')"
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for evaluation results after submission (default: wait for results)"
    )
    
    args = parser.parse_args()
    main(args)
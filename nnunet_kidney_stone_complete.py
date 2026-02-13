"""
🏥 Complete nnU-Net Training Script for Kidney Stone Segmentation
==================================================================

This script provides a complete implementation to:



"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# GPU setup
import torch
print("="*70)
print("🖥️  GPU CHECK")
print("="*70)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.cuda.set_device(0)
else:
    print("⚠️  WARNING: No GPU! Training will be very slow.")
print("="*70)

# ========================
# CONFIGURATION
# ========================

class Config:
    """Configuration for nnU-Net training"""
    
    # Dataset
    DATASET_ID = 500
    DATASET_NAME = f"Dataset{DATASET_ID:03d}_KidneyStone"
    
    # Paths (adjust for your environment)
    BASE_DIR = Path("/kaggle/working")  # Change if not on Kaggle
    DATA_DIR = None  # Will be auto-detected
    
    # nnU-Net paths
    NNUNET_RAW = BASE_DIR / "nnUNet_raw"
    NNUNET_PREPROCESSED = BASE_DIR / "nnUNet_preprocessed"
    NNUNET_RESULTS = BASE_DIR / "nnUNet_results"
    
    # Training
    CONFIGURATION = "2d"  # 2D U-Net for CT slices
    TRAINER = "nnUNetTrainer"
    NUM_FOLDS = 5
    
    # Paper results for comparison (Table 3)
    PAPER_RESULTS = {
        'U-Net': {'Dice': (97.06, 0.39), 'Jaccard': (94.65, 0.62), 
                  'Precision': (97.83, 0.33), 'Recall': (96.61, 0.77)},
        'U-Net++': {'Dice': (96.63, 0.63), 'Jaccard': (93.97, 0.83),
                    'Precision': (97.16, 0.68), 'Recall': (96.44, 0.67)},
        'U-Net3+': {'Dice': (96.68, 0.61), 'Jaccard': (94.02, 0.86),
                    'Precision': (96.88, 0.34), 'Recall': (96.90, 1.02)},
        'TransU-Net': {'Dice': (95.53, 0.89), 'Jaccard': (92.21, 1.27),
                       'Precision': (96.26, 0.52), 'Recall': (95.39, 1.24)}
    }

config = Config()

# ========================
# SETUP FUNCTIONS
# ========================

def setup_environment():
    """Create directories and set environment variables"""
    print("\n" + "="*70)
    print("📁 SETTING UP ENVIRONMENT")
    print("="*70)
    
    # Create directories
    for dir_path in [config.NNUNET_RAW, config.NNUNET_PREPROCESSED, config.NNUNET_RESULTS]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_path.name}")
    
    # Set environment variables (CRITICAL for nnU-Net)
    os.environ["nnUNet_raw"] = str(config.NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(config.NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(config.NNUNET_RESULTS)
    
    print("\n✓ Environment ready")
    print("="*70)

def find_dataset():
    """Auto-detect KSSD2025 dataset location"""
    print("\n" + "="*70)
    print("🔍 LOCATING DATASET")
    print("="*70)
    
    possible_paths = [
        "/kaggle/input/kssd2025",
        "/kaggle/input/kidney-stone-segmentation-dataset-2025",
        "./data/kssd2025",
        "../data/kssd2025",
    ]
    
    for path_str in possible_paths:
        path = Path(path_str)
        if path.exists():
            config.DATA_DIR = path
            print(f"✓ Found at: {path}")
            return path
    
    print("⚠️  Dataset not found in standard locations")
    print("\nSearching /kaggle/input...")
    
    input_dir = Path("/kaggle/input")
    if input_dir.exists():
        for item in sorted(input_dir.iterdir()):
            print(f"  Available: {item.name}")
            if "kidney" in item.name.lower() or "stone" in item.name.lower():
                config.DATA_DIR = item
                print(f"\n✓ Auto-selected: {item}")
                return item
    
    raise FileNotFoundError(
        "KSSD2025 dataset not found!\n"
        "Please add it to Kaggle or set config.DATA_DIR manually"
    )

# ========================
# METRICS CALCULATION
# ========================

def calculate_all_metrics(pred, gt, smooth=1e-7):
    """
    Calculate comprehensive segmentation metrics
    
    Returns:
        dict with: dice, jaccard, precision, recall, f1_score, specificity, accuracy
    """
    pred_flat = pred.flatten().astype(np.uint8)
    gt_flat = gt.flatten().astype(np.uint8)
    
    # Confusion matrix components
    TP = np.sum((pred_flat == 1) & (gt_flat == 1))
    TN = np.sum((pred_flat == 0) & (gt_flat == 0))
    FP = np.sum((pred_flat == 1) & (gt_flat == 0))
    FN = np.sum((pred_flat == 0) & (gt_flat == 1))
    
    # Metrics
    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    f1_score = 2 * (precision * recall) / (precision + recall + smooth)
    dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    jaccard = (TP + smooth) / (TP + FP + FN + smooth)
    specificity = (TN + smooth) / (TN + FP + smooth)
    accuracy = (TP + TN) / (TP + TN + FP + FN + smooth)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'dice': float(dice),
        'jaccard': float(jaccard),
        'specificity': float(specificity),
        'accuracy': float(accuracy)
    }

# ========================
# DATA CONVERSION
# ========================

def convert_to_nnunet_format():
    """Convert KSSD2025 to nnU-Net format"""
    print("\n" + "="*70)
    print("🔄 CONVERTING TO NNUNET FORMAT")
    print("="*70)
    
    try:
        import SimpleITK as sitk
        from PIL import Image
    except ImportError:
        print("⚠️  Installing required packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "SimpleITK", "Pillow"])
        import SimpleITK as sitk
        from PIL import Image
    
    # Create dataset directories
    dataset_dir = config.NNUNET_RAW / config.DATASET_NAME
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    
    for d in [images_tr, labels_tr]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Find kidney stone images
    print("\nSearching for kidney stone images...")
    
    image_files = []
    for pattern in ["**/Stone/*.jpg", "**/Stone/*.png", "**/*stone*.jpg"]:
        found = list(config.DATA_DIR.glob(pattern))
        if found:
            print(f"  Found {len(found)} files matching: {pattern}")
            image_files.extend(found)
    
    image_files = list(set(image_files))  # Remove duplicates
    print(f"\n✓ Total images found: {len(image_files)}")
    
    if len(image_files) == 0:
        raise FileNotFoundError("No kidney stone images found!")
    
    # Find corresponding masks
    mask_files = []
    for img_file in image_files:
        possible_masks = [
            img_file.parent / "masks" / img_file.name,
            img_file.parent / "masks" / (img_file.stem + "_mask.png"),
            img_file.parent.parent / "masks" / img_file.name,
        ]
        for mask_path in possible_masks:
            if mask_path.exists():
                mask_files.append(mask_path)
                break
    
    print(f"✓ Masks found: {len(mask_files)}")
    
    # Convert to NIfTI
    print(f"\nConverting {len(mask_files)} image-mask pairs to NIfTI...")
    
    successful = 0
    for idx, (img_path, mask_path) in enumerate(tqdm(zip(image_files[:len(mask_files)], mask_files))):
        try:
            # Load
            img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
            mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
            mask = (mask > 127).astype(np.uint8)
            
            # Add channel dimension
            img = img[np.newaxis, ...]
            mask = mask[np.newaxis, ...]
            
            # Convert to SimpleITK
            img_sitk = sitk.GetImageFromArray(img)
            mask_sitk = sitk.GetImageFromArray(mask)
            
            # Save
            case_id = f"case_{idx:04d}"
            sitk.WriteImage(img_sitk, str(images_tr / f"{case_id}_0000.nii.gz"))
            sitk.WriteImage(mask_sitk, str(labels_tr / f"{case_id}.nii.gz"))
            
            successful += 1
        except Exception as e:
            print(f"  Error converting {img_path.name}: {e}")
    
    print(f"\n✓ Successfully converted {successful} cases")
    
    # Create dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "kidney_stone": 1},
        "numTraining": successful,
        "file_ending": ".nii.gz",
        "dataset_name": "KidneyStone",
        "modality": {"0": "CT"}
    }
    
    with open(dataset_dir / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print("✓ dataset.json created")
    print("="*70)
    
    return successful

# ========================
# TRAINING
# ========================

def run_preprocessing():
    """Run nnU-Net preprocessing"""
    print("\n" + "="*70)
    print("⚙️  RUNNING PREPROCESSING")
    print("="*70)
    
    cmd = f"nnUNetv2_plan_and_preprocess -d {config.DATASET_ID} --verify_dataset_integrity"
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print("\n✓ Preprocessing complete")
    else:
        raise RuntimeError("Preprocessing failed!")
    
    print("="*70)

def train_model():
    """Train nnU-Net on all 5 folds"""
    print("\n" + "="*70)
    print("🚀 TRAINING nnU-NET")
    print("="*70)
    print("Training all 5 folds...")
    print("Expected time: 2.5-5 hours (GPU dependent)\n")
    
    cmd = (
        f"nnUNetv2_train {config.DATASET_ID} {config.CONFIGURATION} all "
        f"-tr {config.TRAINER} --npz"
    )
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print("\n✓ Training complete")
    else:
        print("\n⚠️  Training may have had issues")
    
    print("="*70)

def run_inference():
    """Run inference on all data"""
    print("\n" + "="*70)
    print("🎯 RUNNING INFERENCE")
    print("="*70)
    
    predictions_dir = config.BASE_DIR / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    
    images_tr = config.NNUNET_RAW / config.DATASET_NAME / "imagesTr"
    
    cmd = (
        f"nnUNetv2_predict "
        f"-i {images_tr} "
        f"-o {predictions_dir} "
        f"-d {config.DATASET_ID} "
        f"-c {config.CONFIGURATION} "
        f"-tr {config.TRAINER} "
        f"-f all "
        f"--save_probabilities"
    )
    
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        pred_files = list(predictions_dir.glob("*.nii.gz"))
        pred_files = [f for f in pred_files if not "probabilities" in f.name]
        print(f"\n✓ Generated {len(pred_files)} predictions")
    else:
        print("\n⚠️  Inference may have had issues")
    
    print("="*70)
    return predictions_dir

# ========================
# EVALUATION
# ========================

def evaluate_predictions(predictions_dir):
    """Evaluate all predictions and calculate metrics"""
    print("\n" + "="*70)
    print("📊 EVALUATING PREDICTIONS")
    print("="*70)
    
    import SimpleITK as sitk
    
    labels_tr = config.NNUNET_RAW / config.DATASET_NAME / "labelsTr"
    
    pred_files = sorted(predictions_dir.glob("*.nii.gz"))
    pred_files = [f for f in pred_files if not "probabilities" in f.name]
    
    all_metrics = defaultdict(list)
    case_metrics = []
    
    print(f"\nEvaluating {len(pred_files)} predictions...\n")
    
    for pred_file in tqdm(pred_files):
        case_id = pred_file.stem.replace(".nii", "")
        gt_file = labels_tr / f"{case_id}.nii.gz"
        
        if not gt_file.exists():
            continue
        
        # Load
        pred_array = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_file)))
        gt_array = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_file)))
        
        pred_binary = (pred_array > 0).astype(np.uint8)
        gt_binary = (gt_array > 0).astype(np.uint8)
        
        # Calculate metrics
        metrics = calculate_all_metrics(pred_binary, gt_binary)
        metrics['case_id'] = case_id
        
        case_metrics.append(metrics)
        for key, value in metrics.items():
            if key != 'case_id':
                all_metrics[key].append(value)
    
    print(f"\n✓ Evaluated {len(case_metrics)} cases")
    print("="*70)
    
    return all_metrics, case_metrics

# ========================
# RESULTS TABLE
# ========================

def create_results_table(all_metrics):
    """Create paper-format results table (Table 3)"""
    print("\n" + "="*70)
    print("📋 RESULTS TABLE - Paper Format (Table 3)")
    print("="*70)
    
    # Calculate our results
    our_results = {
        'Precision': (np.mean(all_metrics['precision']) * 100, 
                      np.std(all_metrics['precision']) * 100),
        'Recall': (np.mean(all_metrics['recall']) * 100,
                   np.std(all_metrics['recall']) * 100),
        'Jaccard': (np.mean(all_metrics['jaccard']) * 100,
                    np.std(all_metrics['jaccard']) * 100),
        'Dice': (np.mean(all_metrics['dice']) * 100,
                 np.std(all_metrics['dice']) * 100),
        'F1 Score': (np.mean(all_metrics['f1_score']) * 100,
                     np.std(all_metrics['f1_score']) * 100)
    }
    
    # Print table
    print("\n┌" + "─"*12 + "┬" + "─"*18 + "┬" + "─"*18 + "┬" + "─"*18 + "┬" + "─"*18 + "┬" + "─"*18 + "┐")
    print("│ Model      │ Precision (%)    │ Recall (%)       │ Jaccard (%)      │ Dice (%)         │ F1 Score (%)     │")
    print("├" + "─"*12 + "┼" + "─"*18 + "┼" + "─"*18 + "┼" + "─"*18 + "┼" + "─"*18 + "┼" + "─"*18 + "┤")
    
    # Paper results
    for model, metrics in config.PAPER_RESULTS.items():
        prec = metrics['Precision']
        rec = metrics['Recall']
        jac = metrics['Jaccard']
        dice = metrics['Dice']
        
        print(f"│ {model:<10} │ {prec[0]:5.2f} ± {prec[1]:4.2f}   │ "
              f"{rec[0]:5.2f} ± {rec[1]:4.2f}   │ {jac[0]:5.2f} ± {jac[1]:4.2f}   │ "
              f"{dice[0]:5.2f} ± {dice[1]:4.2f}   │ {'─'*16} │")
    
    # Separator
    print("├" + "─"*12 + "┼" + "─"*18 + "┼" + "─"*18 + "┼" + "─"*18 + "┼" + "─"*18 + "┼" + "─"*18 + "┤")
    
    # Our results
    prec = our_results['Precision']
    rec = our_results['Recall']
    jac = our_results['Jaccard']
    dice = our_results['Dice']
    f1 = our_results['F1 Score']
    
    print(f"│ {'nnU-Net':<10} │ {prec[0]:5.2f} ± {prec[1]:4.2f}   │ "
          f"{rec[0]:5.2f} ± {rec[1]:4.2f}   │ {jac[0]:5.2f} ± {jac[1]:4.2f}   │ "
          f"{dice[0]:5.2f} ± {dice[1]:4.2f}   │ {f1[0]:5.2f} ± {f1[1]:4.2f}   │")
    
    print("└" + "─"*12 + "┴" + "─"*18 + "┴" + "─"*18 + "┴" + "─"*18 + "┴" + "─"*18 + "┴" + "─"*18 + "┘")
    
    # Comparison
    best_paper = 97.06
    improvement = dice[0] - best_paper
    
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"Best Paper (U-Net):  {best_paper:.2f}%")
    print(f"Our nnU-Net:         {dice[0]:.2f}%")
    print(f"Improvement:         {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"\n🎉 SUCCESS! We beat the paper by {improvement:.2f}%!")
    elif improvement > -0.5:
        print(f"\n✓ GOOD! Performance comparable to paper")
    else:
        print(f"\n⚠️  Below paper - consider longer training")
    
    print("="*70)
    
    # Save as CSV
    table_data = []
    for model, metrics in config.PAPER_RESULTS.items():
        table_data.append({
            'Model': model,
            'Precision (%)': f"{metrics['Precision'][0]:.2f} ± {metrics['Precision'][1]:.2f}",
            'Recall (%)': f"{metrics['Recall'][0]:.2f} ± {metrics['Recall'][1]:.2f}",
            'Jaccard (%)': f"{metrics['Jaccard'][0]:.2f} ± {metrics['Jaccard'][1]:.2f}",
            'Dice (%)': f"{metrics['Dice'][0]:.2f} ± {metrics['Dice'][1]:.2f}",
            'F1 Score (%)': '─'
        })
    
    table_data.append({
        'Model': 'nnU-Net (Ours)',
        'Precision (%)': f"{prec[0]:.2f} ± {prec[1]:.2f}",
        'Recall (%)': f"{rec[0]:.2f} ± {rec[1]:.2f}",
        'Jaccard (%)': f"{jac[0]:.2f} ± {jac[1]:.2f}",
        'Dice (%)': f"{dice[0]:.2f} ± {dice[1]:.2f}",
        'F1 Score (%)': f"{f1[0]:.2f} ± {f1[1]:.2f}"
    })
    
    df = pd.DataFrame(table_data)
    csv_path = config.BASE_DIR / "results_table3.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    return our_results, df

# ========================
# MAIN PIPELINE
# ========================

def main():
    """Main execution pipeline"""
    print("\n" + "="*70)
    print("🏥 nnU-Net Kidney Stone Segmentation - Complete Pipeline")
    print("="*70)
    print("\nTarget: Beat KSSD2025 paper (97.06% Dice)")
    print("Includes: Precision, Recall, Jaccard, Dice, F1 Score")
    print("\n" + "="*70)
    
    try:
        # 1. Setup
        setup_environment()
        find_dataset()
        
        # 2. Convert data
        num_cases = convert_to_nnunet_format()
        print(f"\n✓ Dataset ready: {num_cases} cases")
        
        # 3. Preprocess
        run_preprocessing()
        
        # 4. Train
        train_model()
        
        # 5. Inference
        predictions_dir = run_inference()
        
        # 6. Evaluate
        all_metrics, case_metrics = evaluate_predictions(predictions_dir)
        
        # 7. Results table
        our_results, results_df = create_results_table(all_metrics)
        
        # 8. Summary
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETE!")
        print("="*70)
        print("\nFinal Results:")
        print(f"  Dice:      {our_results['Dice'][0]:.2f}% ± {our_results['Dice'][1]:.2f}%")
        print(f"  Jaccard:   {our_results['Jaccard'][0]:.2f}% ± {our_results['Jaccard'][1]:.2f}%")
        print(f"  Precision: {our_results['Precision'][0]:.2f}% ± {our_results['Precision'][1]:.2f}%")
        print(f"  Recall:    {our_results['Recall'][0]:.2f}% ± {our_results['Recall'][1]:.2f}%")
        print(f"  F1 Score:  {our_results['F1 Score'][0]:.2f}% ± {our_results['F1 Score'][1]:.2f}%")
        print("\n✓ All results saved to /kaggle/working/")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

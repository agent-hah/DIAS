#!/bin/bash

# ==============================================================================
# 1. DEFINE YOUR CHECKPOINT PATHS HERE
# Replace the placeholders with the actual paths to your trained .pth files
# ==============================================================================

# Fully-Supervised Model
FSL_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/UNet_260519_182052/best_model.pth"

# Semi-Supervised Model
SSL_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/FR_UNet_260519_184750/best_model.pth"

# Weakly-Supervised Single Models (Uses standard wsl_test.py)
WSL_PCE_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/FR_UNet_scribble_260521_001405/best_model.pth"
WSL_EM_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/FR_UNet_scribble_260520_154529/best_model.pth"
WSL_RLOSS_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/FR_UNet_scribble_260520_173040/best_model.pth"
WSL_IVM_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/FR_UNet_scribble_260520_194029/best_model.pth"

# Weakly-Supervised Dual Models (Uses wsl_test_doubel_model.py)
WSL_SSCR_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/UNet_DP_scribble_260521_025759/best_model.pth"
WSL_SSCR_ABLATION_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/scribble_FR_UNet_260521_152430/best_model.pth"
WSL_EMA_SSCR_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/scribble_FR_UNet_260520_004032/best_model.pth"

# Weakly-Supervised DMPLS Model (Uses wsl_test_DMPLS.py)
WSL_DMPLS_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/UNet_CCT_scribble_260519_232902/best_model.pth"


# ==============================================================================
# 2. RUN EVALUATIONS
# ==============================================================================

echo "Starting Evaluation Pipeline..."
echo "---------------------------------------------------"

# Fully Supervised
echo "[1/4] Running Fully-Supervised (FSL) Evaluation..."
python full_supervised_segmentation/fsl_test.py -mp "$FSL_CKPT"
echo "---------------------------------------------------"

# Semi-Supervised
echo "[2/4] Running Semi-Supervised (SSL) Evaluation..."
python semi_supervised_segmentation/ssl_test.py -mp "$SSL_CKPT"
echo "---------------------------------------------------"

# Weakly-Supervised (Standard Single Models)
echo "[3/4] Running Weakly-Supervised (Standard) Evaluations..."
python weak_supervised_segmentation/wsl_test.py -mp "$WSL_PCE_CKPT"
python weak_supervised_segmentation/wsl_test.py -mp "$WSL_EM_CKPT"
python weak_supervised_segmentation/wsl_test.py -mp "$WSL_RLOSS_CKPT"
python weak_supervised_segmentation/wsl_test.py -mp "$WSL_IVM_CKPT"
echo "---------------------------------------------------"

# Weakly-Supervised (Dual Models & DMPLS)
echo "[4/4] Running Weakly-Supervised (Dual & DMPLS) Evaluations..."
python weak_supervised_segmentation/wsl_test_doubel_model.py -mp "$WSL_SSCR_CKPT"
python weak_supervised_segmentation/wsl_test_doubel_model.py -mp "$WSL_SSCR_ABLATION_CKPT"
python weak_supervised_segmentation/wsl_test_doubel_model.py -mp "$WSL_EMA_SSCR_CKPT"

python weak_supervised_segmentation/wsl_test_DMPLS.py -mp "$WSL_DMPLS_CKPT"
echo "---------------------------------------------------"

echo "All evaluations complete! Check the save_results/ directory for your CSV files and images."
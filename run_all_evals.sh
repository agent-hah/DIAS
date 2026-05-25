#!/bin/bash

# ==============================================================================
# 1. DEFINE YOUR CHECKPOINT PATHS HERE
# Replace the placeholders with the actual paths to your trained .pth files
# ==============================================================================

# ---> Define your single output directory for all CSV results here <---
FINAL_CSV_DIR="/content/drive/MyDrive/DIAS_Project/results"

# Fully-Supervised Model
FSL_VSS_NET_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/VSS_Net/VSS_Net_NN_260522_162423"
FSL_ATT_UNET_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/Att_UNet/Att_UNet_NN_260522_170758"
FSL_CSNET_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/CSNet/CSNet_NN_260522_173125"
FSL_UNET_NESTED_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/UNet_Nested/UNet_Nested_NN_260522_175415"
FSL_RES_UNET_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/Res_UNet/Res_UNet_NN_260522_181835"
FSL_UNET_3D_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/UNet_3D/UNet_3D_NN_260522_184128"
FSL_FR_UNET_3D_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/FR_UNet_3D/FR_UNet_3D_NN_260522_200920"
FSL_CSNET_3D_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/CSNet_3D/CSNet_3D_NN_260524_230224"
FSL_ATT_UNET_3D_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/Att_UNet_3D/Att_UNet_3D_NN_260522_214842"
FSL_RES_UNET_3D_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/Res_UNet_3D/Res_UNet_3D_NN_260522_223511"
FSL_UNET_NESTED_3D_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/UNet_Nested_3D/UNet_Nested_3D_NN_260522_231250"
FSL_PSC_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/PSC/PSC_NN_260523_004438"
FSL_SVS_NET_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/SVS_Net/SVS_Net_NN_260523_011332"
FSL_MAA_NET_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/MAA_Net/"
FSL_FR_UNET_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/FR_UNet/FR_UNet_NN_260525_060312"
FSL_IPN_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/IPN/IPN_NN_260525_143852"
FSL_ST_UNET_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/ST_UNet/ST_UNet_NN_260525_080225"

# Semi-Supervised Model
SSL_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/FR_UNet_260519_184750/ite_1_student"

# Weakly-Supervised Single Models (Uses standard wsl_test.py)
WSL_PCE_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/FR_UNet_scribble_260521_001405"
WSL_EM_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/FR_UNet_scribble_260520_154529"
WSL_RLOSS_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/FR_UNet_scribble_260520_173040"
WSL_IVM_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/FR_UNet_scribble_260520_194029"

# Weakly-Supervised Dual Models (Uses wsl_test_doubel_model.py)
WSL_SSCR_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/UNet_DP_scribble_260521_025759"
WSL_SSCR_ABLATION_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/scribble_FR_UNet_260521_152430"
WSL_EMA_SSCR_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/scribble_FR_UNet_260520_004032"

# Weakly-Supervised DMPLS Model (Uses wsl_test_DMPLS.py)
WSL_DMPLS_CKPT="/content/drive/MyDrive/DIAS_Project/saved_models/UNet_CCT_scribble_260519_232902"

# ==============================================================================
# 2. RUN EVALUATIONS
# ==============================================================================

echo "Starting Evaluation Pipeline..."
echo "---------------------------------------------------"

# Fully Supervised
echo "[1/4] Running Fully-Supervised (FSL) Evaluation..."
python full_supervised_segmentation/fsl_test.py -mp "$FSL_VSS_NET_CKPT"
python full_supervised_segmentation/fsl_test.py -mp "$FSL_ATT_UNET_CKPT"
python full_supervised_segmentation/fsl_test.py -mp "$FSL_CSNET_CKPT"
python full_supervised_segmentation/fsl_test.py -mp "$FSL_UNET_NESTED_CKPT"
python full_supervised_segmentation/fsl_test.py -mp "$FSL_RES_UNET_CKPT"
python full_supervised_segmentation/fsl_test.py -mp "$FSL_UNET_3D_CKPT"
python full_supervised_segmentation/fsl_test.py -mp "$FSL_FR_UNET_3D_CKPT"
python full_supervised_segmentation/fsl_test.py -mp "$FSL_CSNET_3D_CKPT"
python full_supervised_segmentation/fsl_test.py -mp "$FSL_ATT_UNET_3D_CKPT"
python full_supervised_segmentation/fsl_test.py -mp "$FSL_RES_UNET_3D_CKPT"
python full_supervised_segmentation/fsl_test.py -mp "$FSL_UNET_NESTED_3D_CKPT"
python full_supervised_segmentation/fsl_test.py -mp "$FSL_PSC_CKPT"
python full_supervised_segmentation/fsl_test.py -mp "$FSL_SVS_NET_CKPT"

python full_supervised_segmentation/fsl_train.py -mt MAA_Net --tag NN -wm offline --opts SAVE_DIR "$FSL_MAA_NET_CKPT"
python full_supervised_segmentation/fsl_train.py -mt FR_UNet --tag NN -wm offline --opts SAVE_DIR "$FSL_FR_UNET_CKPT"
python full_supervised_segmentation/fsl_train.py -mt IPN --tag NN -wm offline --opts SAVE_DIR "$FSL_IPN_CKPT"
python full_supervised_segmentation/fsl_train.py -mt ST_UNet --tag NN -wm offline --opts SAVE_DIR "$FSL_ST_UNET_CKPT"
echo "---------------------------------------------------"

# Semi-Supervised
# echo "[2/4] Running Semi-Supervised (SSL) Evaluation..."
# python semi_supervised_segmentation/ssl_test.py -mp "$SSL_CKPT"
# echo "---------------------------------------------------"
#
# # Weakly-Supervised (Standard Single Models)
# echo "[3/4] Running Weakly-Supervised (Standard) Evaluations..."
# python weak_supervised_segmentation/wsl_test.py -mp "$WSL_PCE_CKPT"
# python weak_supervised_segmentation/wsl_test.py -mp "$WSL_EM_CKPT"
# python weak_supervised_segmentation/wsl_test.py -mp "$WSL_RLOSS_CKPT"
# python weak_supervised_segmentation/wsl_test.py -mp "$WSL_IVM_CKPT"
# echo "---------------------------------------------------"
#
# # Weakly-Supervised (Dual Models & DMPLS)
# echo "[4/4] Running Weakly-Supervised (Dual & DMPLS) Evaluations..."
# python weak_supervised_segmentation/wsl_test_doubel_model.py -mp "$WSL_SSCR_CKPT"
# python weak_supervised_segmentation/wsl_test_doubel_model.py -mp "$WSL_SSCR_ABLATION_CKPT"
# python weak_supervised_segmentation/wsl_test_doubel_model.py -mp "$WSL_EMA_SSCR_CKPT"
#
# python weak_supervised_segmentation/wsl_test_DMPLS.py -mp "$WSL_DMPLS_CKPT"
# echo "---------------------------------------------------"
#
# echo "All evaluations complete! Check the save_results/ directory for your CSV files and images."
#
#
# # ==============================================================================
# # 3. CONSOLIDATE RESULTS
# # ==============================================================================
# echo "Consolidating CSV files..."
#
# # Create the final directory if it doesn't already exist
# mkdir -p "$FINAL_CSV_DIR"
#
# # Find all generated CSV files in the save_results directory and copy them to the final directory
# find save_results/ -type f -name "*.csv" -exec cp {} "$FINAL_CSV_DIR/" \;
#
# echo "All evaluations complete!"
# echo "Your consolidated CSV results have been saved to: $FINAL_CSV_DIR"


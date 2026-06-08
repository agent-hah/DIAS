#!/bin/bash

# Clear previous output file if it exists
>output.txt

BASE_DIR="/home/ashmithandoo/projects/lab/saved_models"
FSL_BASE_DIR="$BASE_DIR/fully_supervised"
SSL_BASE_DIR="$BASE_DIR/semi_supervised"
WSL_BASE_DIR="$BASE_DIR/weakly_supervised"

# Helper function to run the evaluation and redirect output to BOTH terminal and file
run_eval() {
  local script_path=$1
  local model_path=$2
  shift 2 # Shift the first two arguments so $@ contains any extra flags

  echo "---------------------------------------------------" | tee -a output.txt
  echo "Evaluating: $model_path" | tee -a output.txt
  if [ $# -gt 0 ]; then
    echo "Extra arguments: $@" | tee -a output.txt
  fi

  # Run the script, pass any extra arguments ($@), combine stdout and stderr (2>&1), and pipe to tee
  python "$script_path" -mp "$model_path" "$@" 2>&1 | tee -a output.txt
}

echo "Starting Evaluation Pipeline... All logs will be saved to output.txt AND shown here." | tee -a output.txt
echo "===================================================" | tee -a output.txt

# ==============================================================================
# 1. FULLY-SUPERVISED EVALUATIONS
# ==============================================================================
echo "[1/3] Running Fully-Supervised (FSL) Evaluations..." | tee -a output.txt

run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/FR_UNet/FR_UNet_NN_260529_184205"
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/Att_UNet/Att_UNet_NN_260527_222210"
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/CSNet/CSNet_NN_260527_231159"

# ---> Turned off AMP for IPN and MAA_Net <---
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/IPN/IPN_NN_260606_041238" --opts AMP False
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/MAA_Net/MAA_Net_NN_260606_084342" --opts AMP False

run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/PSC/PSC_NN_260528_220236"
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/ST_UNet/ST_UNet_NN_260529_194151"
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/SVS_Net/SVS_Net_NN_260528_230109"
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/VSS_Net/VSS_Net_NN_260527_210416"
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/CSNet_3D/CSNet_3D_NN_260528_122908"
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/UNet_Nested/UNet_Nested_NN_260528_000323"
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/UNet_Nested_3D/UNet_Nested_3D_NN_260528_195427"
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/UNet_3D/UNet_3D_NN_260528_021342"
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/UNet/UNet_NN_260529_175103"
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/Res_UNet/Res_UNet_NN_260528_011013"
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/Res_UNet_3D/Res_UNet_3D_NN_260528_182747"
run_eval "full_supervised_segmentation/fsl_test.py" "$FSL_BASE_DIR/FR_UNet_3D/FR_UNet_3D_NN_260528_094634"

# ==============================================================================
# 2. SEMI-SUPERVISED EVALUATIONS
# ==============================================================================
echo "" | tee -a output.txt
echo "[2/3] Running Semi-Supervised (SSL) Evaluations..." | tee -a output.txt

# Define the models and iterations to loop through (pseudo_label folders removed)
SSL_MODELS=(
  "1_60/FR_UNet_260531_205724"
  "10_30_SDA/FR_UNet_260531_074329"
  "3_60/FR_UNet_260601_161037"
  "3_60_SDA/FR_UNet_260602_020818"
  "3_30_SDA/FR_UNet_260530_162356"
  "1_30_SDA/FR_UNet_260530_043833"
  "10_60/FR_UNet_260602_094212"
  "1_60_SDA/FR_UNet_260601_060734"
  "10_60_SDA/FR_UNet_260602_190000"
)

ITERATIONS=("ite_1_teacher" "ite_1_student" "ite_2_student" "ite_3_student")

# Loop over all requested iterations (teachers and students) for all configs
for model in "${SSL_MODELS[@]}"; do
  for iteration in "${ITERATIONS[@]}"; do
    target_dir="$SSL_BASE_DIR/$model/$iteration"
    if [ -d "$target_dir" ]; then
      run_eval "semi_supervised_segmentation/ssl_test.py" "$target_dir"
    fi
  done
done

# ==============================================================================
# 3. WEAKLY-SUPERVISED EVALUATIONS
# ==============================================================================
echo "" | tee -a output.txt
echo "[3/3] Running Weakly-Supervised (WSL) Evaluations..." | tee -a output.txt

# --- SALE Models ---
run_eval "weak_supervised_segmentation/wsl_test.py" "$WSL_BASE_DIR/SALE/wsl_train_entropy_mini/UNet_SALE_260604_124157"
run_eval "weak_supervised_segmentation/wsl_test.py" "$WSL_BASE_DIR/SALE/wsl_train_GatedCRFLoss/UNet_SALE_260604_141900"
run_eval "weak_supervised_segmentation/wsl_test.py" "$WSL_BASE_DIR/SALE/wsl_train_Inter&Intra_Class/UNet_SALE_260604_160148"
run_eval "weak_supervised_segmentation/wsl_test.py" "$WSL_BASE_DIR/SALE/wsl_train_pcedice/UNet_SALE_260604_110725"
run_eval "weak_supervised_segmentation/wsl_test_DMPLS.py" "$WSL_BASE_DIR/SALE/wsl_train_DMPLS/UNet_CCT_SALE_260606_230638"

run_eval "weak_supervised_segmentation/wsl_test_doubel_model.py" "$WSL_BASE_DIR/SALE/wsl_train_sscr/UNet_SALE_260605_022144"
run_eval "weak_supervised_segmentation/wsl_test_doubel_model.py" "$WSL_BASE_DIR/SALE/wsl_train_sscr_ablation/SALE_UNet_260605_051721"

# EMA_sscr Fix
run_eval "weak_supervised_segmentation/wsl_test_EMA.py" "$WSL_BASE_DIR/SALE/wsl_train_EMA_sscr/SALE_UNet_260605_194816"

# --- RDFA Models ---
run_eval "weak_supervised_segmentation/wsl_test.py" "$WSL_BASE_DIR/RDFA/wsl_train_entropy_mini/UNet_RDFA_260603_210412"
run_eval "weak_supervised_segmentation/wsl_test.py" "$WSL_BASE_DIR/RDFA/wsl_train_GatedCRFLoss/UNet_RDFA_260603_223747"
run_eval "weak_supervised_segmentation/wsl_test.py" "$WSL_BASE_DIR/RDFA/wsl_train_Inter&Intra_Class/UNet_RDFA_260604_001144"
run_eval "weak_supervised_segmentation/wsl_test.py" "$WSL_BASE_DIR/RDFA/wsl_train_pcedice/UNet_RDFA_260603_193335"
run_eval "weak_supervised_segmentation/wsl_test_DMPLS.py" "$WSL_BASE_DIR/RDFA/wsl_train_DMPLS/UNet_CCT_RDFA_260606_185011"

run_eval "weak_supervised_segmentation/wsl_test_doubel_model.py" "$WSL_BASE_DIR/RDFA/wsl_train_sscr/UNet_RDFA_260604_014249"
run_eval "weak_supervised_segmentation/wsl_test_doubel_model.py" "$WSL_BASE_DIR/RDFA/wsl_train_sscr_ablation/RDFA_UNet_260604_062445"

# EMA_sscr Fix
run_eval "weak_supervised_segmentation/wsl_test_EMA.py" "$WSL_BASE_DIR/RDFA/wsl_train_EMA_sscr/RDFA_UNet_260605_153256"

echo "===================================================" | tee -a output.txt
echo "All evaluations complete! Logs have been saved to output.txt." | tee -a output.txt

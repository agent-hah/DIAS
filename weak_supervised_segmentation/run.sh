#!/bin/bash

# Base directory for your custom saves
BASE_DIR="$HOME/projects/lab/saved_models/weakly_supervised"

# Array to keep track of failed experiments
FAILED_EXPERIMENTS=()

# python weak_supervised_segmentation/wsl_train_sscr.py -mt UNet -st "SALE" --opts SAVE_DIR "" --resume "$BASE_DIR/SALE/wsl_train_sscr/UNet_SALE_260604_173752"

# List of all the weakly supervised training scripts
TRAIN_SCRIPTS=(
  # "wsl_train_pcedice.py"
  # "wsl_train_entropy_mini.py"
  # "wsl_train_GatedCRFLoss.py"
  # "wsl_train_Inter&Intra_Class.py"
  "wsl_train_DMPLS.py"
  # "wsl_train_sscr.py"
  # "wsl_train_EMA_sscr.py"
  # "wsl_train_sscr_ablation.py"
)

# List of scribble types to evaluate
SCRIBBLE_TYPES=(
  "RDFA"
  "SALE"
)

# Function to execute a training run
run_experiment() {
  local script_name=$1
  local scribble_type=$2

  # Remove the '.py' extension to create a clean folder name for the save directory
  local model_name="${script_name%.py}"

  # Split the save directory by scribble type
  local save_path="${BASE_DIR}/${scribble_type}/${model_name}"

  echo "========================================================="
  echo "Running WSL Model: $model_name"
  echo "Scribble Type: $scribble_type"
  echo "Executing Script: weak_supervised_segmentation/$script_name"
  echo "Save Directory: $save_path"
  echo "========================================================="

  # Run the script from the root directory, passing the scribble type and custom save path
  python "weak_supervised_segmentation/$script_name" -mt UNet_CCT -st "$scribble_type" --opts SAVE_DIR "$save_path"
  # Capture the exit code of the python command
  local exit_code=$?
  # Check if the command failed (a non-zero exit code indicates an error)
  if [ $exit_code -ne 0 ]; then
    echo ""
    echo "-> ERROR: $script_name with $scribble_type failed with exit code $exit_code."
    echo "-> Recording failure and continuing to the next experiment..."

    # Add the failed configuration to our tracking array
    FAILED_EXPERIMENTS+=("$script_name | Type: $scribble_type (Exit Code: $exit_code)")
  fi
}

# Loop through the scribble types, then loop through the scripts
for stype in "${SCRIBBLE_TYPES[@]}"; do
  for script in "${TRAIN_SCRIPTS[@]}"; do
    run_experiment "$script" "$stype"
  done
done

echo ""
echo "========================================================="
# Check if the FAILED_EXPERIMENTS array has any items in it
if [ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]; then
  echo "SUCCESS: All weakly supervised experiments completed without any errors!"
else
  echo "ATTENTION: The following WSL experiments failed during execution:"
  echo "---------------------------------------------------------"
  # Loop through the array and print each recorded failure
  for failed_run in "${FAILED_EXPERIMENTS[@]}"; do
    echo " - $failed_run"
  done
  echo "---------------------------------------------------------"
  # Exit with a non-zero status so the terminal knows the script had errors
  exit 1
fi
echo "========================================================="

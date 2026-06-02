#!/bin/bash

# Base directory for your custom saves
BASE_DIR="$HOME/projects/lab/saved_models/semi_supervised"

# Array to keep track of failed experiments
FAILED_EXPERIMENTS=()

# Function to execute a training run
run_experiment() {
  local nl=$1
  local nu=$2
  local use_sda=$3
  local save_path=""

  if [ "$use_sda" = true ]; then
    save_path="${BASE_DIR}/${nl}_${nu}_SDA"
    echo "========================================================="
    echo "Running Labeled: $nl | Unlabeled: $nu | SDA: ON"
    echo "Save Directory: $save_path"
    echo "========================================================="
    python semi_supervised_segmentation/ssl_train.py -nl "$nl" -nu "$nu" --SDA --opts SAVE_DIR "$save_path"
  else
    save_path="${BASE_DIR}/${nl}_${nu}"
    echo "========================================================="
    echo "Running Labeled: $nl | Unlabeled: $nu | SDA: OFF"
    echo "Save Directory: $save_path"
    echo "========================================================="
    python semi_supervised_segmentation/ssl_train.py -nl "$nl" -nu "$nu" --opts SAVE_DIR "$save_path"
  fi

  # Capture the exit code of the python command
  local exit_code=$?

  # Check if the command failed (a non-zero exit code indicates an error)
  if [ $exit_code -ne 0 ]; then
    echo ""
    echo "-> ERROR: This experiment failed with exit code $exit_code."
    echo "-> Recording failure and continuing to the next experiment..."

    # Add the failed configuration to our tracking array
    FAILED_EXPERIMENTS+=("Labeled: $nl | Unlabeled: $nu | SDA: $use_sda (Exit Code: $exit_code)")
  fi
}

# --- LABELED DATA: 1 SEQUENCE ---
# run_experiment 1 30 true
# run_experiment 1 60 false
# run_experiment 1 60 true

# --- LABELED DATA: 3 SEQUENCES ---
# run_experiment 3 30 true
# run_experiment 3 60 false
python semi_supervised_segmentation/ssl_train.py -nl 3 -nu 60 --opts SAVE_DIR "$BASE_DIR/3_60" --resume "$BASE_DIR/3_60/FR_UNet_260601_161037/ite_1_student" --start_ite 1 --start_phase student
run_experiment 3 60 true

# --- LABELED DATA: 10 SEQUENCES ---
# run_experiment 10 30 true
run_experiment 10 60 false
run_experiment 10 60 true

echo ""
echo "========================================================="
# Check if the FAILED_EXPERIMENTS array has any items in it
if [ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]; then
  echo "SUCCESS: All experiments completed without any errors!"
else
  echo "ATTENTION: The following experiments failed during execution:"
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

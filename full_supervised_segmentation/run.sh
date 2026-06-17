#!/bin/bash
python full_supervised_segmentation/fsl_train.py -mt VSS_Net --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/fully_supervised/VSS_Net"
# python full_supervised_segmentation/fsl_train.py -mt Att_UNet --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/Att_UNet"
# python full_supervised_segmentation/fsl_train.py -mt CSNet --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/CSNet"
# python full_supervised_segmentation/fsl_train.py -mt UNet_Nested --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/UNet_Nested"
# python full_supervised_segmentation/fsl_train.py -mt Res_UNet --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/Res_UNet"
# python full_supervised_segmentation/fsl_train.py -mt UNet_3D --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/UNet_3D"
# python full_supervised_segmentation/fsl_train.py -mt FR_UNet_3D --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/FR_UNet_3D"
# python full_supervised_segmentation/fsl_train.py -mt CSNet_3D --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/fully_supervised/CSNet_3D" --resume "/home/ashmithandoo/projects/lab/saved_models/fully_supervised/CSNet_3D/CSNet_3D_NN_260528_122908"
# python full_supervised_segmentation/fsl_train.py -mt Res_UNet_3D --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/fully_supervised/Res_UNet_3D"
# python full_supervised_segmentation/fsl_train.py -mt UNet_Nested_3D --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/fully_supervised/UNet_Nested_3D"
# python full_supervised_segmentation/fsl_train.py -mt PSC --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/fully_supervised/PSC"
# python full_supervised_segmentation/fsl_train.py -mt SVS_Net --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/fully_supervised/SVS_Net" --resume "/home/ashmithandoo/projects/lab/saved_models/fully_supervised/SVS_Net/SVS_Net_NN_260528_230109"
# python full_supervised_segmentation/fsl_train.py -mt IPN --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/fully_supervised/IPN" AMP False
# python full_supervised_segmentation/fsl_train.py -mt UNet --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/fully_supervised/UNet"
# python full_supervised_segmentation/fsl_train.py -mt MAA_Net --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/fully_supervised/MAA_Net" AMP False
# python full_supervised_segmentation/fsl_train.py -mt FR_UNet --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/fully_supervised/FR_UNet"
# python full_supervised_segmentation/fsl_train.py -mt ST_UNet --tag NN -wm offline --opts SAVE_DIR "$HOME/projects/lab/saved_models/fully_supervised/ST_UNet"

# Array to store the names of any models that fail
# failed_models=()
#
# # List of models to train
# models=("MAA_Net" "IPN")
#
# for model in "${models[@]}"; do
#   echo "=================================================="
#   echo "Starting training for $model..."
#
#   # Run the command directly within the if statement.
#   # The '!' checks if the command failed (returned a non-zero exit code).
#   if ! python full_supervised_segmentation/fsl_train.py -mt "$model" --tag NN -wm offline --opts SAVE_DIR "/content/drive/MyDrive/DIAS_Project/saved_models/$model"; then
#     echo "❌ ERROR: $model encountered an exception!"
#     failed_models+=("$model")
#   else
#     echo "✅ SUCCESS: $model completed."
#   fi
# done
#
# # Print the final summary
# echo ""
# echo "================ SUMMARY ================"
# if [ ${#failed_models[@]} -eq 0 ]; then
#   echo "🎉 All models trained successfully!"
# else
#   echo "⚠️ The following models FAILED:"
#   for failed in "${failed_models[@]}"; do
#     echo " - $failed"
#   done
# fi
#

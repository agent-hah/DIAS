import torch
import argparse
import os


def check_trained_epoch(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Error: The file '{checkpoint_path}' does not exist.")
        return

    try:
        # Load the checkpoint dictionary
        # Using map_location='cpu' ensures it works even on a machine without a GPU
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract the epoch
        if "epoch" in checkpoint:
            epoch = checkpoint["epoch"]
            print(
                f"Success! The model at '{checkpoint_path}' was trained up to epoch: {epoch}"
            )

            # You can also optionally print out the best monitor metric if it exists
            if "monitor_best" in checkpoint:
                print(f"Best monitor metric achieved: {checkpoint['monitor_best']}")
        else:
            print(
                f"The checkpoint at '{checkpoint_path}' does not contain an 'epoch' key."
            )

    except Exception as e:
        print(f"Failed to load checkpoint. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check the epoch a semi-supervised segmentation model was trained to."
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the checkpoint file (e.g., best_model.pth or final_checkpoint.pth)",
    )

    args = parser.parse_args()
    check_trained_epoch(args.checkpoint)

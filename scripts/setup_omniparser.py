import os
import sys
import shutil

try:
    from huggingface_hub import snapshot_download
except Exception:
    print("Error: huggingface_hub is required. Install with `pip install huggingface-hub`.")
    sys.exit(1)


def setup():
    # project_root is the parent of the scripts/ folder
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(project_root, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Target folder name expected by the code
    target_caption_folder = os.path.join(weights_dir, "icon_caption_florence")

    # if model exists, do nothing
    safetensors_path = os.path.join(target_caption_folder, "model.safetensors")
    if os.path.exists(safetensors_path):
        print(f"✅ Models found in {target_caption_folder}. Skipping download.")
        return

    print("⬇️  Models missing. Downloading OmniParser-v2 snapshot into weights/ ...")

    try:
        snapshot_download(
            repo_id="microsoft/OmniParser-v2.0",
            revision=None,
            local_dir=weights_dir,
            allow_patterns=["icon_detect/*", "icon_caption/*"],
        )
    except Exception as e:
        print("❌ Failed to download from HuggingFace Hub:", str(e))
        print("If the repo is private, set HUGGINGFACE_HUB_TOKEN in your environment or run `huggingface-cli login`.")
        sys.exit(1)

    downloaded_caption = os.path.join(weights_dir, "icon_caption")
    if os.path.exists(downloaded_caption) and not os.path.exists(target_caption_folder):
        print(f"🔄 Renaming {downloaded_caption} -> {target_caption_folder} to match code expectations...")
        try:
            os.rename(downloaded_caption, target_caption_folder)
        except Exception as e:
            print("❌ Rename failed:", e)
            sys.exit(1)

    print("✅ Download and layout complete.")


if __name__ == "__main__":
    setup()

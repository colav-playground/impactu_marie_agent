"""
Download optimized models for 4GB VRAM GPU.

Models selected for efficiency and quality:
- Qwen2-1.5B-Instruct: Best balance of quality/size
- Phi-2: Strong reasoning capabilities
- SmolLM-1.7B: Fast inference
"""

from huggingface_hub import snapshot_download
import os

# Models optimized for 4GB VRAM
MODELS = [
    {
        "name": "Qwen/Qwen2-1.5B-Instruct",
        "description": "Best quality for size, excellent instruction following",
        "size": "~3GB",
        "priority": 1
    },
    {
        "name": "microsoft/phi-2",
        "description": "Strong reasoning, good for thinking tasks",
        "size": "~5.5GB",  
        "priority": 2
    },
    {
        "name": "HuggingFaceTB/SmolLM-1.7B-Instruct",
        "description": "Fast inference, optimized",
        "size": "~3.4GB",
        "priority": 3
    }
]

def download_model(model_id: str, cache_dir: str = "./models"):
    """Download model from HuggingFace."""
    print(f"\n{'='*80}")
    print(f"Downloading: {model_id}")
    print(f"{'='*80}\n")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=os.path.join(cache_dir, model_id.replace("/", "--")),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"\n✓ Downloaded: {model_id}\n")
        return True
    except Exception as e:
        print(f"\n✗ Error downloading {model_id}: {e}\n")
        return False

def main():
    """Download all models."""
    print("\n" + "="*80)
    print("MARIE Agent - Model Download")
    print("Optimized for 4GB VRAM GPU")
    print("="*80 + "\n")
    
    cache_dir = os.path.expanduser("~/.cache/marie_models")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Models will be downloaded to: {cache_dir}\n")
    
    for model in sorted(MODELS, key=lambda x: x["priority"]):
        print(f"Model: {model['name']}")
        print(f"Description: {model['description']}")
        print(f"Size: {model['size']}")
        
        response = input(f"\nDownload {model['name']}? [Y/n]: ").strip().lower()
        
        if response in ['', 'y', 'yes']:
            success = download_model(model['name'], cache_dir)
            if success:
                print(f"✓ {model['name']} ready to use")
        else:
            print(f"Skipped {model['name']}")
    
    print("\n" + "="*80)
    print("Download complete!")
    print(f"Models location: {cache_dir}")
    print("\nTo use with vLLM:")
    print("  export HF_HOME=" + cache_dir)
    print("  python -m marie_agent.cli 'your query'")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

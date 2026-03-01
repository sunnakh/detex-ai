from datasets import load_dataset
import pandas as pd
from pathlib import Path

DATASET_NAME = "artem9k/ai-text-detection-pile"
OUTPUT_DIR = Path(__file__).resolve().parent.parent /"data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Downloading dataset from Hugging Face...")
    
    try:
        dataset = load_dataset(DATASET_NAME)
    except Exception as e:
        print(f"Failed to download the dataset: {e}")

    print(dataset)
    
    for split in dataset.keys():
        df = dataset[split].to_pandas()
        output_path = OUTPUT_DIR / f"{split}.csv"
        df.to_csv(output_path, index= False)
        print(f"Saved {split} -> {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()

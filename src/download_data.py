from datasets import load_dataset
import pandas as pd
from pathlib import Path

DATASET_NAME = "artem9k/ai-text-detection-pile"
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    
    print("Downloading dataset from Hugging Face")
    dataset = load_dataset(name= DATASET_NAME)
    
    print(dataset)
    
    for split in dataset.keys():
        
        df = dataset[split].to_pandas()
        
        output_path = OUTPUT_DIR / f"{split}.csv"
        df.to_csv(output_path, index= False)
        

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download HotPotQA data for CEP experiment
"""

import os
import requests
import json
from tqdm import tqdm

def download_file(url: str, filename: str):
    """Download a file with progress bar"""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # HotPotQA data URLs
    data_urls = {
        "hotpot_dev_distractor_v1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "hotpot_dev_fullwiki_v1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
        "hotpot_train_v1.1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
    }
    
    # Download files
    for filename, url in data_urls.items():
        filepath = os.path.join("data", filename)
        if not os.path.exists(filepath):
            try:
                download_file(url, filepath)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists, skipping...")
    
    # Verify data format
    print("\nVerifying data format...")
    for filename in data_urls.keys():
        filepath = os.path.join("data", filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"{filename}: {len(data)} samples")
                
                # Show sample structure
                if len(data) > 0:
                    sample = data[0]
                    print(f"  Sample keys: {list(sample.keys())}")
                    print(f"  Question: {sample['question'][:100]}...")
                    print(f"  Answer: {sample['answer']}")
                    print(f"  Context paragraphs: {len(sample['context'])}")
                    print(f"  Supporting facts: {len(sample['supporting_facts'])}")
                    print()
            except Exception as e:
                print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    main() 
"""
HotPotQA Dataset Implementation
"""

import json
import os
import requests
from typing import List, Optional, Dict
from dataclasses import dataclass
from tqdm import tqdm

from shared import Dataset, Problem

@dataclass
class HotPotQASample:
    """Data structure for a HotPotQA sample"""
    _id: str
    question: str
    answer: str
    supporting_facts: List[List[str]]
    context: List[List[str]]  # List of [title, sentences]
    type: Optional[str] = None
    level: Optional[str] = None

class HotpotDataset(Dataset):
    """HotPotQA dataset implementation"""
    
    def __init__(self):
        self.data_folder = "data/hotpot"
        # Map of domain names to filenames
        self.domain_to_file = {
            "hotpot_train": "hotpot_train_v1.1.json",
            "hotpot_dev_distractor": "hotpot_dev_distractor_v1.json", 
            "hotpot_dev_fullwiki": "hotpot_dev_fullwiki_v1.json"
        }
        self._ensure_data_exists()
    
    def _ensure_data_exists(self):
        """Check if data exists and download if not"""
        for domain, filename in self.domain_to_file.items():
            filepath = os.path.join(self.data_folder, filename)
            if not os.path.exists(filepath):
                self._download_data(domain, filename)
    
    def _download_data(self, domain: str, filename: str):
        """Download HotPotQA data for a specific domain"""
        # Create data directory if it doesn't exist
        os.makedirs(self.data_folder, exist_ok=True)
        
        # HotPotQA data URLs
        data_urls = {
            "hotpot_train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
            "hotpot_dev_distractor": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
            "hotpot_dev_fullwiki": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
        }
        
        # Get URL for the domain
        if domain in data_urls:
            url = data_urls[domain]
            filepath = os.path.join(self.data_folder, filename)
            self._download_file(url, filepath)
        else:
            raise FileNotFoundError(f"No download URL found for domain {domain}")
    
    def _download_file(self, url: str, filename: str):
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
        
        print(f"Successfully downloaded {filename}")
    
    def get_dataset_name(self) -> str:
        return "hotpot"
    
    def get_domains(self) -> List[str]:
        return list(self.domain_to_file.keys())
    
    def get_problems(self, domain: str, max_samples: int) -> List[Problem]:
        if domain not in self.domain_to_file:
            raise ValueError(f"HotPotQA dataset only supports {list(self.domain_to_file.keys())} domains, got: {domain}")
        
        filename = self.domain_to_file[domain]
        data_file = os.path.join(self.data_folder, filename)
        
        # Load data using same approach as original
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        problems = []
        for item in data[:max_samples]:
            # Format context using same approach as original
            context = self.format_context(item['context'])
            
            problem = Problem(
                question=item['question'],
                context=context,
                gold_answer=item['answer'],
                problem_id=item['_id']
            )
            problems.append(problem)
        
        return problems
    
    def format_context(self, context: List[List[str]]) -> str:
        """Format context for prompt - same as original experiment"""
        formatted_context = []
        for title, sentences in context:
            formatted_context.append(f"Title: {title}")
            formatted_context.extend(sentences)
            formatted_context.append("")
        return "\n".join(formatted_context) 
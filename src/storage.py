import json
import os
import pandas as pd
from datetime import datetime

DATA_FILE = "data/dataset.jsonl"

def ensure_data_dir():
    if not os.path.exists("data"):
        os.makedirs("data")

def save_interaction(prompt, original_ai_response, corrected_response, comments, tags):
    """
    Saves a DPO-ready training example.
    
    Structure:
    - Prompt: The input
    - Chosen: Your 'Perfect' version (The correction)
    - Rejected: The AI's 'Draft' version (The original)
    """
    ensure_data_dir()
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "chosen": corrected_response,  # The Good Data (Expert)
        "rejected": original_ai_response, # The Bad Data (AI)
        "meta": {
            "comments": comments,
            "tags": tags,
            "annotator": "Expert_v1"
        }
    }
    
    # Append to JSONL file
    with open(DATA_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    return True

def load_stats():
    """Returns basic stats about the dataset size"""
    if not os.path.exists(DATA_FILE):
        return 0, pd.DataFrame()
    
    data = []
    with open(DATA_FILE, "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    return len(data), df
import os
import json
import pandas as pd
from typing import List, Dict, Any
import numpy as np

# Define the data directory
DATA_DIR = "data"
ANALYSIS_FILE = os.path.join(DATA_DIR, "analyses.json")

def ensure_data_directory():
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def save_analysis(analysis: Dict[str, Any]):
    """
    Save an analysis to persistent storage.
    
    Args:
        analysis: Dictionary containing analysis results
    """
    ensure_data_directory()
    
    # Load existing analyses
    analyses = load_previous_analyses()
    
    # Add the new analysis
    analyses.append(analysis)
    
    # Save to file
    with open(ANALYSIS_FILE, 'w', encoding='utf-8') as f:
        # Convert numpy arrays to lists for JSON serialization
        json.dump(analyses, f, ensure_ascii=False, indent=2)

def load_previous_analyses() -> List[Dict[str, Any]]:
    """
    Load previous analyses from storage.
    
    Returns:
        List of analysis dictionaries
    """
    ensure_data_directory()
    
    # Check if file exists
    if not os.path.exists(ANALYSIS_FILE):
        return []
    
    try:
        # Load the analyses from file
        with open(ANALYSIS_FILE, 'r', encoding='utf-8') as f:
            analyses = json.load(f)
        
        return analyses
    except Exception as e:
        print(f"Error loading analyses: {e}")
        return []

def export_to_csv(analysis: Dict[str, Any], filename: str):
    """
    Export an analysis to CSV format.
    
    Args:
        analysis: Dictionary containing analysis results
        filename: Name of the output file
    """
    ensure_data_directory()
    
    # Create a dataframe for responses and clusters
    df = pd.DataFrame({
        'question': [analysis['question']] * len(analysis['responses']),
        'response': analysis['responses'],
        'cluster': [f"Cluster {c+1}" for c in analysis['clusters']]
    })
    
    # Save to CSV
    output_path = os.path.join(DATA_DIR, filename)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    return output_path

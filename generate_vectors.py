#!/usr/bin/env python3
"""
Generate vector embeddings for medical record text chunks.
Uses sentence-transformers with a small efficient model.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def main():
    # Load the medical record JSON file
    print("Loading medical record data...")
    try:
        with open('medical_record.json', 'r') as f:
            medical_texts = json.load(f)
        print(f"Loaded {len(medical_texts)} text chunks")
    except FileNotFoundError:
        print("Error: medical_record.json not found!")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in medical_record.json!")
        return
    
    # Initialize the sentence transformer model
    # Using all-MiniLM-L6-v2: small (22MB), fast, and effective
    print("Loading sentence transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Generate embeddings for all text chunks
    print("Generating embeddings...")
    embeddings = model.encode(X
        medical_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalize for better similarity search
    )
    
    # Save embeddings to .npy file
    output_file = 'medical_record_vectors.npy'
    print(f"Saving embeddings to {output_file}...")
    np.save(output_file, embeddings)
    
    # Print summary information
    print(f"\nCompleted successfully!")
    print(f"- Text chunks processed: {len(medical_texts)}")
    print(f"- Embedding dimensions: {embeddings.shape}")
    print(f"- Output file: {output_file}")
    print(f"- File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    
    # Verify that indices match
    print(f"\nIndex verification:")
    print(f"- JSON array length: {len(medical_texts)}")
    print(f"- Embeddings array shape: {embeddings.shape[0]} vectors")
    print(f"- Indices match: {len(medical_texts) == embeddings.shape[0]}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
"""
Download FMA and GTZAN datasets
"""
import os
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import shutil

def download_file(url, dest_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))
    
    return dest_path

def download_fma():
    """Download FMA Medium dataset"""
    fma_path = Path('data/fma_large')
    fma_path.mkdir(parents=True, exist_ok=True)
    
    # Download FMA Medium (25GB)
    # Note: This is large; for demo, use a smaller subset
    fma_url = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
    zip_path = fma_path / 'fma_medium.zip'
    
    if not zip_path.exists():
        print("Downloading FMA Medium (this will take time)...")
        download_file(fma_url, zip_path)
    
    # Extract
    if not (fma_path / 'fma_medium').exists():
        print("Extracting FMA Medium...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(fma_path)
    
    # Download metadata
    meta_url = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
    meta_path = fma_path / 'fma_metadata.zip'
    
    if not meta_path.exists():
        print("Downloading metadata...")
        download_file(meta_url, meta_path)
    
    if not (fma_path / 'fma_metadata').exists():
        print("Extracting metadata...")
        with zipfile.ZipFile(meta_path, 'r') as zip_ref:
            zip_ref.extractall(fma_path)
    
    print("FMA dataset ready")

def download_gtzan():
    """Download GTZAN dataset"""
    gtzan_path = Path('data/gtzan')
    gtzan_path.mkdir(parents=True, exist_ok=True)
    
    # GTZAN URL (from Kaggle)
    # For simplicity, use a mirror or direct download
    gtzan_url = "https://github.com/karolpiczak/GTZAN-dataset/archive/refs/heads/master.tar.gz"
    tar_path = gtzan_path / 'gtzan.tar.gz'
    
    if not tar_path.exists():
        print("Downloading GTZAN...")
        download_file(gtzan_url, tar_path)
    
    # Extract
    if not (gtzan_path / 'gtzan-master').exists():
        print("Extracting GTZAN...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(gtzan_path)
    
    print("GTZAN dataset ready")

def main():
    print("Starting data download...")
    
    # Create data directory
    Path('data').mkdir(exist_ok=True)
    
    # Download datasets
    download_gtzan()
    
    # For FMA, you might want to use a smaller subset to save time
    print("For FMA, consider using a smaller subset or skipping for demo")
    # download_fma()
    
    print("Data download complete!")

if __name__ == "__main__":
    main()

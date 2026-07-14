#!/usr/bin/env python
"""Setup script"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aura-music",
    version="1.0.0",
    author="sswfml",
    description="Mood-based music discovery using vector search and deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sswfml/DLS_project",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "faiss-cpu>=1.7.4",
        "pymilvus>=2.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "mlflow>=2.5.0",
        "wandb>=0.15.0",
        "gradio>=3.50.0",
        "streamlit>=1.25.0",
        "pytest>=7.4.0",
        "black>=23.0.0",
        "flake8>=6.1.0",
    ],
    entry_points={
        "console_scripts": [
            "run-experiments=scripts.run_experiments:main",
            "download-data=scripts.download_data:main",
        ],
    },
)

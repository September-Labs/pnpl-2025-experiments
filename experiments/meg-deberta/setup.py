"""
Setup script for MEG-DeBERTa Classifier
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="meg-deberta-classifier",
    version="0.1.0",
    author="September Labs",
    description="MEG phoneme classification using DeBERTa attention and focal loss",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/September-Labs/pnpl-2025-experiments",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",  # CC BY-NC 4.0
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "lightning>=2.0.0",
        "torchmetrics>=1.0.0",
        "numpy>=1.24.0",
        "h5py>=3.0.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.10",
            "flake8>=5.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "tensorboard>=2.11.0",
            "wandb>=0.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "meg-train=scripts.train:main",
            "meg-evaluate=scripts.evaluate:main",
            "meg-submit=scripts.generate_submission:main",
        ],
    },
    keywords="MEG, EEG, phoneme, classification, deep learning, DeBERTa, attention",
)
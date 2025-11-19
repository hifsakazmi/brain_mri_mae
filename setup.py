# /content/brain_mri_mae/setup.py
from setuptools import setup, find_packages

setup(
    name="brain_mri_mae",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision", 
        "matplotlib",
        "Pillow",
        "kagglehub",
    ],
)
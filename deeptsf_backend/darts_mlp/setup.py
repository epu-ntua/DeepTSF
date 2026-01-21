from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="darts-mlp",
    version="0.2.0",
    author="nickbrakis",
    author_email="nibrakis@gmail.com",
    description="Multi-Layer Perceptron (MLP) model for Darts time series forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nickbrakis/darts-mlp",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "u8darts==0.28.0", 
        "torch==2.2.2",
        "numpy==1.26.4",
        "pytorch-lightning==2.2.1",
        "torchmetrics==1.3.2",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
)

# Regular setup.py file for the package
from setuptools import setup, find_packages

setup(
    name="lex",
    version="0.1",
    packages=["lex"],
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "sentencepiece",
        "tiktoken",
        "tqdm",
        "numpy",
        "matplotlib",
        "pandas",
        "jupyterlab"
    ],
)

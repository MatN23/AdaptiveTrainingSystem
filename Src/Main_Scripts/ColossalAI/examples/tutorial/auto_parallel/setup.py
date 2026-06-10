# Copyright (c) 2025 MatN23. All rights reserved.
from setuptools import find_packages, setup

setup(
    name="auto_parallel",
    version="0.0.1",
    description="",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
    ],
)
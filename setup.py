import os

from setuptools import find_packages, setup

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read long description
with open("README.md") as f:
    long_description = f.read()

setup(
    name="CLIP_HAR_PROJECT",
    version="0.1.0",
    description="Human Action Recognition using CLIP models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="tuandung12092002",
    author_email="tuandung12092002@gmail.com",
    url="https://github.com/tuandung12092002/CLIP_HAR_PROJECT",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
            "mypy",
            "pytest",
            "pytest-cov",
        ],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "clip-har-train=CLIP_HAR_PROJECT.train:main",
            "clip-har-evaluate=CLIP_HAR_PROJECT.evaluate:main",
            "clip-har-serve=CLIP_HAR_PROJECT.mlops.inference_serving:main",
        ],
    },
)

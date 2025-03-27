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
    description="Human Action Recognition using CLIP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="tuandunghcmut",
    author_email="tuandunghcmut@example.com",
    url="https://github.com/tuandunghcmut/CLIP_HAR_PROJECT",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "clip-har-train=CLIP_HAR_PROJECT.train:main",
            "clip-har-evaluate=CLIP_HAR_PROJECT.evaluate:main",
            "clip-har-serve=CLIP_HAR_PROJECT.mlops.inference_serving:main",
        ],
    },
)

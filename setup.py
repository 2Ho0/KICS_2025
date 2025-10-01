#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="dpr",
    version="1.0.0",
    description="Facebook AI Research Open Domain Q&A Toolkit",
    url="https://github.com/facebookresearch/DPR/",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["dpr*", "conf*"]),  # ✅ 명시적으로 둘 다 포함
    include_package_data=True,                           # ✅ 데이터 동반 설치
    setup_requires=[
        "setuptools>=18.0",
    ],
    install_requires=[
        "faiss-cpu>=1.6.1",
        "filelock",
        "numpy",
        "regex",
        "torch>=1.5.0",
        "transformers>=4.3",
        "tqdm>=4.27",
        "wget",
        "spacy>=2.1.8",
        "hydra-core>=1.0.0",
        "omegaconf>=2.0.1",
        "jsonlines",
        "soundfile",
        "editdistance",
    ],
)

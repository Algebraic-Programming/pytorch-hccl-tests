#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "pandas==1.3.5",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="HCCL Test Authors",
    author_email="hccl-tests@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="End-to-end PyTorch distributed benchamrks (HCCL backend)",
    entry_points={
        "console_scripts": [
            "torch-hccl-benchs=pytorch_hccl_tests.cli:main",
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="pytorch_hccl_tests",
    name="pytorch_hccl_tests",
    packages=find_packages(include=["pytorch_hccl_tests", "pytorch_hccl_tests.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/Algebraic-Programming/pytorch-hccl-tests",
    version="0.1.15",
    zip_safe=False,
)

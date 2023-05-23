#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "torch==1.11.0+cpu",
    "torchvision==0.15.0+cpu",
    "pydot==1.4.2",
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="E2E HCCL PyTorch tests",
    entry_points={
        "console_scripts": [
            "pytorch_hccl_tests=pytorch_hccl_tests.cli:main",
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
    url="https://github.com/zouzias/pytorch_hccl_tests",
    version="0.1.0",
    zip_safe=False,
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="direct-optimizer",
    version="0.1.0",
    author="Alberto Uliana",
    author_email="uliana.2047848@studenti.uniroma1.it",
    description="A high-performance, vectorized implementation of the DIRECT algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    packages=find_packages(),
    
    extras_require={
        "dev": ["pytest>=6.0"],
    },
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
    ],
    
    python_requires=">=3.7",
    
    install_requires=[
        "numpy>=1.18.0",
    ],
    
    license="GPLv3",
)
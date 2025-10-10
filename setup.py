from setuptools import find_packages, setup
import os

# Read the README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Document Forgery Detection using Computer Vision and Machine Learning"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

setup(
    name='document-forgery-detection',
    packages=find_packages(),
    version='1.0.0',
    description='Document Forgery Detection using Computer Vision and Machine Learning',
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author='Ztrimus',
    author_email='contact@document-forgery-detection.com',
    url='https://github.com/Ztrimus/Document-Forgery-Detection',
    license='MIT',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme>=0.5",
            "nbsphinx>=0.8",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "document-forgery-detection=cli:cli",
            "dfd=cli:cli",
        ],
    },
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "document forgery detection",
        "computer vision",
        "machine learning",
        "image processing",
        "fraud detection",
        "artificial intelligence",
        "deep learning",
        "document analysis",
    ],
)

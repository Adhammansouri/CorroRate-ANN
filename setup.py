from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="corrorate-ann",
    version="1.0.0",
    author="Adham Mansouri",
    author_email="your.email@example.com",
    description="Advanced Artificial Neural Network for Corrosion Rate Prediction in MDEA-based Solutions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Adhammansouri/CorroRate-ANN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    keywords="corrosion, neural network, MDEA, prediction, machine learning, chemistry, engineering",
    project_urls={
        "Bug Reports": "https://github.com/Adhammansouri/CorroRate-ANN/issues",
        "Source": "https://github.com/Adhammansouri/CorroRate-ANN",
        "Documentation": "https://github.com/Adhammansouri/CorroRate-ANN#readme",
    },
) 
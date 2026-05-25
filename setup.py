"""Setup script for Lock-In Focus Monitor"""

from setuptools import find_packages, setup
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="lock-in-monitor",
    version="1.0.0",
    author="Adit Rahman",
    description="Real-time focus monitoring with ResNet + PyTorch + OpenCV + Streamlit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adit-rah/lock-in",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Office/Business :: Time Tracking",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "lock-in=src.app:main",
            "lock-in-train=src.train:main",
            "lock-in-dashboard=src.dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.md"],
    },
)

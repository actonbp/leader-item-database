from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="leader-item-database",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A project for analyzing leadership constructs using embedding models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/leader-item-database",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas",
        "numpy",
        "openai",
        "scikit-learn",
        "matplotlib",
        "python-dotenv",
    ],
    extras_require={
        "dev": ["pytest", "flake8"],
    },
)

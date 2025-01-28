from setuptools import setup, find_packages

setup(
    name="dataset",
    version="0.1.0",
    description="A library to fetch and store GitHub issues with labels in SQLite.",
    author="Sharjeel Yunus",
    author_email="sharjeel924@gmail.com",
    packages=find_packages(),
    install_requires=[
        "requests>=2.0.0",
        "langdetect>=1.0.9",
        "python-dotenv>=0.21.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

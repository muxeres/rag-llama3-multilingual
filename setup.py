"""Setup script for RAG Chatbot."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rag-llama3-multilingual",
    version="1.0.0",
    author="RAG Chatbot Team",
    description="Multilingual RAG chatbot with Llama 3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/muxeres/rag-llama3-multilingual",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rag-chatbot=cli:main",
        ],
    },
)

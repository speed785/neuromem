"""
neuromem — Smart Context Manager for LLM agents
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuromem",
    version="0.1.0",
    author="speed785",
    description="Smart Context Manager for LLM agents — intelligent token budgeting, scoring, summarization, and pruning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/speed785/neuromem",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.9",
    install_requires=[
        # core has zero required deps — pure Python stdlib only
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.7.0"],
        "langchain": ["langchain>=0.1.0", "langchain-openai>=0.1.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.7.0",
            "langchain>=0.1.0",
            "langchain-openai>=0.1.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "ruff",
            "mypy",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm context-window token-management langchain openai agents memory",
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="high-speed-fuse-break-analysis",
    version="0.1.0",
    author="Oussama Guelfaa",
    author_email="oussama.guelfaa@example.com",
    description="Analysis of high-speed X-ray radiography videos of industrial fuses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Oussama-Guelfaa/High-Speed-Fuse-Break-Analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)

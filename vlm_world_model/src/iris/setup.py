from setuptools import setup, find_packages

setup(
    name="iris",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ale-py>=0.7.4",
        "einops>=0.3.2",
        "gym[accept-rom-license]>=0.21.0",
        "hydra-core>=1.1.1",
        "opencv-python",
        "protobuf>=3.20.0",
        "psutil>=5.8.0",
        "pygame>=2.1.2",
        "tqdm>=4.66.4",
        "wandb>=0.12.7",
    ],
    python_requires=">=3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for Atari game experiments with reinforcement learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/iris",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
) 
from setuptools import find_packages, setup

setup(
    name="grok",
    #package_dir={"": ".", "torchph": "torchph/torchph"},
    packages=find_packages(include=["grok", "grok.*", "phd", "phd.*", "torchph", "torchph.*"]),
    version="0.0.1",
    install_requires=[
        "pytorch_lightning",
        "blobfile",
        "numpy",
        "torch",
        "tqdm",
        "scipy",
        "mod",
        "matplotlib",
        "torchvision",
    ],
)

from setuptools import setup, find_packages

setup(
    name="ml2048",
    author="SuXYIO",
    version="0.1",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "matplotlib",
        "numpy",
        "pygame",
        "torch",
        "gymnasium",
        "evotorch",
    ],
)

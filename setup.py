from setuptools import setup, find_packages

setup(
    name="ml_richelet",
    version="0.1.0",
    description="Machine Learning Final Project",
    author="RICHELET Arthur",
    packages=find_packages(),     # Automatically finds 'lib' as a package
    include_package_data=True,
    install_requires=[
        "pandas",
        "tensorflow==2.15.0", # Tensorflow version compatible with CUDA 12.4 & CUDNN 8.9
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "keras",
    ],
)

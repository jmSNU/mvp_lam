from setuptools import setup, find_packages

setup(
    name="egoexo4d",
    version="0.1.0",
    packages=find_packages(include=["egoexo4d"]),
    include_package_data=True,
    description="TFDS builder for Ego-Exo4D episodes",
)
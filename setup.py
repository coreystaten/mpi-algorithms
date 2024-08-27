from typing import List

from setuptools import find_packages, setup


def parse_requirements() -> List[str]:
    with open("requirements.txt") as f:
        return f.read().splitlines()


setup(
    name="mpi-algorithms",
    version="0.1.0",
    packages=find_packages(include=["mpi_algorithms", "mpi_algorithms.*"]),
    install_requires=parse_requirements(),
    entry_points={},
)

from setuptools import find_packages, setup

requirements = [
    "numpy>=1.16",
    "fdm",
    "algebra>=0.3.0",
    "plum-dispatch>=0.2.3",
    "backends>=0.4.5",
    "backends-matrix>=0.3.1",
]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)

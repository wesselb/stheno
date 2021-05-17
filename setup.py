from setuptools import find_packages, setup

requirements = [
    "numpy>=1.16",
    "fdm",
    "algebra>=1",
    "plum-dispatch>=1",
    "backends>=1.1.1",
    "backends-matrix>=1.0.3",
    "mlkernels>=0.3",
    "wbml>=0.3.3",
]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)

from setuptools import find_packages, setup

requirements = [
    "numpy>=1.16",
    "fdm",
    "algebra>=1",
    "plum-dispatch>=1.5.3",
    "backends>=1.4.11",
    "backends-matrix>=1.2.3",
    "mlkernels>=0.3.4",
    "wbml>=0.3.3",
]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)

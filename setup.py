from setuptools import setup, find_packages

setup(
    name="fastbandits",
    version="0.1",
    packages=find_packages(exclude=["tests*"]),
    author="Jialin Yi",
    install_requires=[
        # list dependencies here
        "numpy",
    ],
)
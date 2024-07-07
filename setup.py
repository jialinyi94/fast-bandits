from setuptools import setup, find_packages

setup(
    name="xbandits",
    version="0.0.2",
    packages=find_packages(exclude=["tests*"]),
    author="Jialin Yi",
    install_requires=[
        # list dependencies here
        "numpy",
    ],
)
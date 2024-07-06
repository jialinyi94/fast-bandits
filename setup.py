from setuptools import setup, find_packages

setup(
    name="fast-bandits",
    version="0.1",
    packages=find_packages(exclude=["tests*"]),
    author="Jialin Yi",
    install_requires=[
        # list dependencies here
    ],
)
"""This is needed to make 'src' available as a module within the docker container"""

import setuptools

setuptools.setup(
    name="dataflow",
    version="0.1.0",
    install_requires=[],
    packages=setuptools.find_packages(),
)
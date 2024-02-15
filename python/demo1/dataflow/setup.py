"""This is needed to make 'src' available as a module within the docker container"""

import setuptools

setuptools.setup(
    name="dataflow",
    version="0.1.0",
    install_requires=['apache-beam[gcp]==2.54.0rc1',
                      'tensorflow-transform'],
    packages=setuptools.find_packages(),
)
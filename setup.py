import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="wav2lip",
    version="0.0.1",
    author="Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.",
    description="Wav2Lip",
    long_description=read("README.md"),
    url="https://github.com/Rudrabha/Wav2Lip",
    packages=find_packages(),
)

import os.path as osp
from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        content = f.read()
    return content


def find_version():
    version_file = 'torchdet3d/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def get_requirements(filename='requirements.txt'):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


setup(
    name='torchdet3d',
    version=find_version(),
    description='A library for deep learning 3D object detection in PyTorch',
    author='Sovrasov Vladislav, Prokofiev Kirill',
    license='MIT',
    long_description=readme(),
    url='https://github.com/sovrasov/3d-object-detection.pytorch',
    packages=find_packages(),
    install_requires=get_requirements(),
    keywords=['3D object detection', 'Deep Learning', 'Computer Vision'],
)

from setuptools import setup, find_packages

setup(
    name='fourierfun',
    version='0.1.2',
    description='A package for working with Fourier transforms of images and/' 
                'or sequences.',
    url='https://github.com/bram98/fourierfun.git',
    author='Bram Verreussel',
    license='MIT',
    packages=['fourierfun'],
    install_requires=['numpy',
                      'matplotlib',
                      ],
    
    classifiers=[],
)
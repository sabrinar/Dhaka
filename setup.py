# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:46:10 2017

@author: t-sarash
"""

from setuptools import setup, find_packages




setup(name='autoencoder',
      version='0.1',
      description='Autoencoder package for single cell genomic data',
      classifiers=[
        'Development Status :: pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Computational Biology :: Genomics',
      ],
      keywords='Single cell Autoencoders',
      url='https://github.com/MicrosoftGenomics/Dhaka',
      author='Sabrina Rashid',
      author_email='t-sarash@microsoft.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)

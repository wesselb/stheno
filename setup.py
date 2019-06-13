# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup

requirements = ['numpy',
                'scipy<=1.2.1',
                'autograd',
                'tensorflow',
                'torch',

                'fdm',
                'plum-dispatch',
                'backends']

setup(packages=find_packages(exclude=['docs']),
      install_requires=requirements)
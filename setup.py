#!/usr/bin/env python

from setuptools import setup

def read(filename):
    with open(filename,'r') as fd:
        return fd.readlines()

setup(name='paparazzi',
      version='0.1',
      description='Paparazzi Differentiable Renderer',
      author='Derek Liu, Michael Tao, Alec Jacobson',
      author_email='derek@dgp.toronto.edu, mtao@dgp.toronto.edu, jacobson@dgp.toronto.edu',
      url='http://www.dgp.toronto.edu/projects/paparazzi/',
      install_requires=read("./requirements.txt"),
      packages=['paparazzi'],
      )

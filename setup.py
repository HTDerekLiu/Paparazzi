#!/usr/bin/env python

from setuptools import setup

setup(name='paparazzi',
      version='0.1',
      description='Paparazzi Differentiable Renderer',
      author='Derek Liu, Michael Tao, Alec Jacobson',
      author_email='derek@dgp.toronto.edu, mtao@dgp.toronto.edu, jacobson@dgp.toronto.edu',
      url='http://www.dgp.toronto.edu/projects/paparazzi/',
      py_modules=['paparazzi'],
      install_requires=[
          'pyopengl',
          'pyglfw',
          ],
      )

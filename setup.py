#!/usr/bin/env python

from setuptools import setup

install_requires = []
dependency_links = []
with open("./requirements.txt",'r') as fd:
    lines = map(lambda x: x.strip(),fd.readlines())
    for line in lines:
        if line[:2] == "-e":
            link = line.split()[1]
            egg=link.split("#egg=")[1]
            print(link,"====",egg)
            install_requires.append(egg)
            dependency_links.append(link)
        else:
            install_requires.append(line)

setup(name='paparazzi',
      version='0.1',
      description='Paparazzi Differentiable Renderer',
      author='Derek Liu, Michael Tao, Alec Jacobson',
      author_email='derek@dgp.toronto.edu, mtao@dgp.toronto.edu, jacobson@dgp.toronto.edu',
      url='http://www.dgp.toronto.edu/projects/paparazzi/',
      install_requires=install_requires,
      dependency_links=dependency_links,
      packages=['paparazzi'],
      )

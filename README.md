# Paparazzi
Paparazzi: Surface Editing by way of Multi-view Image Processing

#### Installation
Paparazzi has two versions available: one (this master branch) has a more "Matlab-esque" main loop. This version can be installed by simply cloning the repository and following the intructions below to install the dependencies. The other one is a more [modular version](https://github.com/HTDerekLiu/Paparazzi/tree/modular) that separates components of our main loop. The modular version Paparazzi can be installed as a ```pip``` installable Python2 library:
```bash
pip2 install git+https://github.com/HTDerekLiu/Paparazzi.git@modular
```

#### Dependencies
Paparazzi depends on [PyElTopo](https://github.com/mtao/pyeltopo), which in turn depends on a fork of [ElTopo](https://github.com/tysonbrochu/eltopo). Although ```pip``` will install these dependencies will automatically be pulled, ElTopo depends on several C/C++ dependencies that ```pip``` cannot handle and must be installed through other means:

Paparazzi is tested on Ubuntu 16.04 machine on python 2.7. Dependencies include Eigen, BLAS, LAPACK. One option to install the dependencies is to run, on an apt-based system (Ubuntu/Debian):
```bash
apt-get install libeigen3-dev libblas-dev liblapack-dev
```
On a portage-based system (Gentoo/Funtoo) this would be
```bash
emerge blas-reference eigen:3 glfwlapack
```

Paparazzi renderer is based on PyOpenGL and PyGLFW. One option to instal the deendencies is to run
```bash
apt-get install libglfw3
apt-get install libglfw3-dev
pip install pyglfw
pip install PyOpenGL
```

#### Bibtex
```
@article{Liu:Paparazzi:2018,
  title = {Paparazzi: Surface Editing by way of Multi-View Image Processing},
  author = {Hsueh-Ti Derek Liu and Michael Tao and Alec Jacobson},
  year = {2018},
  journal = {ACM Transactions on Graphics}, 
}
```

# Paparazzi
Paparazzi: Surface Editing by way of Multi-view Image Processing

#### Installation
Paparazzi can be installed as a ```pip``` installable Python2 library:
```bash
pip2 install git+https://github.com/HTDerekLiu/Paparazzi.git@modular
```

Paparazzi depends on [PyElTopo](https://github.com/mtao/pyeltopo), which in turn depends on a fork of [ElTopo](https://github.com/tysonbrochu/eltopo). Although ```pip``` will install these dependencies will automatically be pulled, ElTopo depends on several C/C++ dependencies that ```pip``` cannot handle and must be installed through other means:

### Dependencies
Paparazzi is tested on Ubuntu 16.04 machine on python 2.7. Dependencies include Eigen, BLAS, LAPACK. One option to install the dependencies is to run, on an apt-based system (Ubuntu/Debian):
```bash
apt-get install libeigen3-dev libblas-dev liblapack-dev libglfw3-dev
```
On a portage-based system (Gentoo/Funtoo) this would be
```bash
emerge blas-reference eigen:3 glfwlapack
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

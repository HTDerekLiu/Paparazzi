# Paparazzi
Paparazzi: Surface Editing by way of Multi-view Image Processing

### Modular version
We have two versions available: one with a "Matlab-esque" main loop in the ```master``` branch and a a more [modular version](https://github.com/HTDerekLiu/Paparazzi/tree/modular) in the ```modular``` branch that separates separates components of our main loop.
The modular branch can be installed installed with ```pip```:
```bash
pip2 install git+https://github.com/HTDerekLiu/Paparazzi.git@modular
```

### Fetching the repo
Please remember to pull the submodules for this repository:
```bash
git submodule update --init --recursive
```

#### Setup
Paparazzi is tested on Ubuntu 16.04 machine on python 2.7. Dependencies include Eigen, BLAS, LAPACK, PyOpenGL, and PyGLFW. One option to install the dependencies is to run
```
sudo apt-get install libeigen3-dev
sudo apt-get install libblas-dev
sudo apt-get install liblapack-dev
pip install PyOpenGL
sudo apt-get install libglfw3
sudo apt-get install libglfw3-dev
pip install pyglfw
```

Paparazzi uses pyeltopo, a tool necessary for mesh cleaning, please run 
```
pip install pyeltopo
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

#### Additional Reference
Hsueh-Ti Derek Liu, Michael Tao, Chun-Liang Li, Derek Nowrouzezahrai, Alec Jacobson, _Beyond Pixel Norm-Balls: Parametric Adversaries using an Analytically Differentiable Renderer_, ICLR 2019

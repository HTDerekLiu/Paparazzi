# Paparazzi
This is the sample code for _Paparazzi: Surface Editing by way of Multi-view Image Processing_.

#### Fetching the repo
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

Paparazzi uses pyeltopo, a tool necessary for mesh cleaning, please run and install it with the `build_pyeltopo.sh` script
```bash
bash build_pyeltopo.sh [-i] [-h] [-p pythonpath]
```
* `-h` to get a help message
* `-i` to install (recommended)
* `-p` to use the python found at `pythonpath`
where `pythonpath` can be found by typing `which python` in the terminal.

#### Bibtex
```
@article{Liu:Paparazzi:2018,
  title = {Paparazzi: Surface Editing by way of Multi-View Image Processing},
  author = {Hsueh-Ti Derek Liu and Michael Tao and Alec Jacobson},
  year = {2018},
  journal = {ACM Transactions on Graphics}, 
}
```

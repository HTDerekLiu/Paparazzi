# Paparazzi
Paparazzi: Surface Editing by way of Multi-view Image Processing

#### Fetching the repo
Please remember to pull the submodules for this repository:
```bash
git submodule update --init --recursive
```

#### Setup
To install pyeltopo, a tool necessary for mesh cleaning, please run and install it with the `build_pyeltopo.sh` script

```bash
bash build_pyeltopo.sh [-i] [-h] [-p pythonpath]
```
* `-h` to get a help message
* `-i` to install (recommended)
* `-p` to use the python found at `pythonpath`

#### Derek Setup
Pyeltop depends on Eigen, BLAS, and LAPACK, which can be installed with
```
sudo apt-get install libeigen3-dev
sudo apt-get install libblas-dev
sudo apt-get install liblapack-dev
```

After installing dependencies, pyeltopo,  a tool necessary for mesh cleaning, can be installed with
```
bash build_pyeltopo.sh -i -h path/to/python
```
where `path/to/python` can be found by typing `which python` in the terminal.

Paparazzi rendering is built on top of OpenGL and GLFW, which can be installed with
```
pip install PyOpenGL
sudo apt-get install libglfw3
sudo apt-get install libglfw3-dev
pip install pyglfw
```

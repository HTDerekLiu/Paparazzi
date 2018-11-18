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

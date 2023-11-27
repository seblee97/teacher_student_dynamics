# Teacher Student Dynamics

This repository contains code related to variations of the teacher-student framework. It can be used to investigate continual learning, transfer learning, critical learning periods among other learning regimes. Results from papers such as

- [Goldt et. al (2019)](https://proceedings.neurips.cc/paper/2019/hash/cab070d53bd0d200746fb852a922064a-Abstract.html)
- [Goldt et al (2020)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.041044)
- [Lee et. al (2020)](https://proceedings.mlr.press/v139/lee21e.html?ref=https://githubhelp.com)
- [Lee et. al (2021)](https://arxiv.org/abs/2205.09029)

should all be reproducible from this code.

## Getting Started

### Python
The majority of the code is written in Python. Any version above Python 3.5 should work, although extensive testing has not been carried out (development was done in Python 3.9 so this will be the most reliable version). Other python requirements can be satisfied by running (preferably in a virtual environment):

```pip install -r requirements.txt```

Finally, the package itself should be installed by running 

```pip install -e .```

from the root of the repository.

### C++
Some parts, specifically the ODEs are implemented in C++ (C++17). I plan to add a python implementation to forgo these additional requirements (see [TODOs](#todos) below). In addition to standard C++17 compiler requirements, the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) package is also needed for the linear algebra computations. Saving the Eigen header files in the root of the repository should be sufficient. See the Eigen [documentation](https://eigen.tuxfamily.org/dox/GettingStarted.html) for more details.

## Basic Usage

The main interface of the code is the ```config.yaml``` file found in the experiments folder. In this file you can specify the parameters of an experiment, which can then be run using ```python run.py```.

A single run will produce a number of output files. By default these will be located in ```experiments/results/``` under a folder named by the timestamp of the experiment. The files will include scalar data (e.g. generalisation errors and order parameters) for the network simulations and/or the ODE solutions, and plots of this data (under a subfolder named _plots_).

## Coverage

Below is a summary of the models / experiment configurations that have been implemented, and those that are planned. For features not yet implemented, the asterisk * denotes that it has been completed but note yet integrated/pushed.

### Implemented

- Standard teacher-student framework with IID Gaussian inputs.
- Hidden Manifold Model (HMM) where input data has non-trivial correlations.
- Multi teacher extensions of the above.
    - Teachers rotated in feature and/or readout space.
    - Identical teachers.
    - Teachers with fraction of nodes shared and fraction rotated.
    - Interpolation between different projection matrices for HMM.
- Interleaved replay of previous teacher during training of second (networks only).
- Output noise to teachers.
- Input noise to student only (e.g. for critical learning)
- Input noise to student only (e.g. for critical learning)
- Frozen hidden units (e.g. for critical learning)

### TODOs

#### _Major_

- Classification (currently only regression is implemented).
- RL Perceptron ([Nish et. al (2023)](https://arxiv.org/abs/2306.10404))

### Not Planned

The following features are not planned but are certainly possible:

- *More than 2 teachers.
- *More than 2 layers.
- *Mean field scaling.
- *MNIST or other arbitrary datasets.
- *Symmetric initialisation for students.
- *Other teacher configurations (e.g. drifting).
- *Copying head at switch.
- *Path integral consolidation (Zenke, Poole).

 Code for most exist already but have not been pushed to this repository in an attempt to minimise complexity. 
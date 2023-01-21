# Teacher Student Dynamics

This repository contains code related to variations of the teacher-student framework. It can be used to investigate continual learning, transfer learning, critical learning periods among other learning regimes.

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

## Coverage

### Implemented

- Vanilla Teacher Student
  - Rotating teachers
  - Identical teachers
  - Shared node teachers
- Interleaved Replay
- Output noise to teachers
- Input noise to students [Networks only]
- 
### TODOs

#### Major TODOs
#### Minor TODOs

- Standardize naming convention from some base reference for ODE and network runner.
- Log overlap / confoguring frequency of logging 
- Consolidate cpp outputs using data logging module from ode runner class
- Logging in network using new numpy array system
- Noise outputs for teacher on runner side
- Plotting
- Interleaving for ODEs
- Frozen units 
- All the config checks on ODE/Network compatibility
- Switch steps in ODEs
- Meta/

Next Implementations

- Classification (previously implemented)
- ODEs for HMM multi-teacher
- Networks for HMM multi-teacher
- EWC for networks (previously implemented)
- EWC ODEs (previously derived but mistake somewhere)
- Equivalent python implementation of ODEs for python-only use-case (implemented previously)

### Not Planned
Not planned but possible

- More than 2 teachers (previously implemented)
- More than 2 layers (previously implemented)
- Mean field scaling (previously implemented)
- MNIST or other arbitrary datasets (previously implemented)
- Symmetric initialisation for students
- Other teacher configurations (drifting)
- Copying head at switch
- Path integral consolidation (Zenke, Poole)
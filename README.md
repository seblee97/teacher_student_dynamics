Implemented

- Vanilla Teacher Student
  - Rotating teachers
  - Identical teachers
  - Shared node teachers
- Interleaved Replay
- Output noise to teachers
- Input noise to students [Networks only]

TODO

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

Not planned but possible

- More than 2 teachers (previously implemented)
- More than 2 layers (previously implemented)
- Mean field scaling (previously implemented)
- MNIST or other arbitrary datasets (previously implemented)
- Symmetric initialisation for students
- Other teacher configurations (drifting)
- Copying head at switch
- Path integral consolidation (Zenke, Poole)
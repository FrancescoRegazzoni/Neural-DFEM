# Neural-DFEM

This repository contains the code required to reproduce the results presented in the paper _Hyperelastic constitutive model discovery with differentiable finite elements and structure-preserving neural networks_.

## Requirements

The code requires `FEniCS`. For installation instructions, please refer to [the official documentation](https://fenicsproject.org/download/archive/). 
In addition, the following Python packages are required (tested versions are indicated):
- `dolfin-adjoint` (2023.0.0)
- `numpy` (1.21.4)
- `pandas` (1.3.4)

To reproduce a tested software environment, you can create a [`conda`](https://docs.conda.io/) environment using the following commands:

```bash
conda create -n neural_dfem_env -c conda-forge \
    python=3.8.12 \
    fenics=2019.1.0 \
    numpy=1.21.4 \
    pandas=1.3.4
conda activate neural_dfem_env
python -m pip install dolfin-adjoint==2023.0.0
```

## Usage

The workflow consists of three main steps:

1. Training data generation  
2. Model training  
3. Model testing

### 1. Training data generation

Training data are generated using a finite element solver. 
The following command computes the displacement fields and global reaction forces for **Setup 1** using the **Mooney–Rivlin** material model.

```bash
./generate_data.py -m config/models/mooneyrivlin.json -t config/setups/setup1.json
```

The generated data are stored in `data/data`.

Additional setups and material models are available in the `config` folder.

### 2. Training

Once the training data are available, the model can be trained.
The following command performs training by specifying:

- the ground-truth material model (`-M`), used to identify the synthetic dataset and for evaluation
- the training setup (`-t`)
- the level of synthetic noise (`--noise`) and random seed  (`-S`)
- the candidate model ansatz (`-m`)
- the maximum number of epochs (`--maxiter`), set here to a small value to keep this demonstration lightweight

```bash
./train.py -t config/setups/setup1.json -d setup1 --noise 1e-3 -S 0 -M config/models/mooneyrivlin.json -m config/models/HNN.json --maxiter 40
```

Additional options can be listed by running `./train.py -h`.
Hyperparameters can be modified in `config/models/HNN.json`.

The script automatically creates a directory under `data/models` containing the training output. The directory name encodes the main training settings.
For instance, in the above case, the output path is the following, which you may store for convenience in a variable:

```bash
export trained_model='data/models/mooney-rivlin_C1_1.000_C2_0.800_K_1.000/_/adjoint_setup1_fb-obs_ns-1.0e-03_HNN-5neur-sp-sp-nsk-stiff10.0-inifact0.1-linear-log-init0-seed2'
```

The folder contains two files:

- `history.json` – training log (loss values, parameters, gradient norm, training time)
- `energy.json` – hyperparameters and trained model parameters

### 3. Testing

After training, the model can be evaluated on different mechanical setups to assess generalization performance.

The following commands generate ground-truth data for **Setup 2** and **Setup 3**. 
The `--export-pvd` flag produces `.pvd` files that can be visualized e.g. in ParaView.

```bash
./generate_data.py -m config/models/mooneyrivlin.json -t config/setups/setup2.json --export-pvd
./generate_data.py -m config/models/mooneyrivlin.json -t config/setups/setup3.json --export-pvd
```

Once the model has been trained, predictions can be computed using the learned energy functional:

```bash
./generate_data.py -m $trained_model/energy.json -t config/setups/setup2.json --export-pvd
./generate_data.py -m $trained_model/energy.json -t config/setups/setup3.json --export-pvd
```

The predicted displacement fields and reaction forces can finally be compared with the corresponding ground-truth data, e.g. by using ParaView.

## Reference

F. Regazzoni, [*Hyperelastic constitutive model discovery with differentiable finite elements and structure-preserving neural networks*](https://arxiv.org/abs/2603.26517), arXiv preprint arXiv:2603.26517, 2026.
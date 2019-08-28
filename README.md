# cnndockbench

### Conda environment

Execute to replicate package environment

```bash
conda env create -f conda_env.yml
```

You can also precompute some data. Install git lfs first and execute:

```bash
git lfs pull
```

### Run model

The model implementation provided here requires:

- A (possibly unprotonated) protein `.pdb` file.
- Ligand SMILES with an `.smi` file.
- Binding pocket center coordinates (__x, y, z__).

One can simply run the pretrained model using the `prod.py` script:

```bash
python prod.py -pdb protein.pdb -smi ligands.smi -x 1.61 -y -47.56 -z 6.82 -output /home/user
```

The script will generate three files: `rmsd_min.csv`, `rmsd_ave.csv` and `n_rmsd.csv` in the `-output` folder or in the current directory if unspecified.

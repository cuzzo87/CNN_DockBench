# cnndockbench

Accompanying code for the manuscript _A deep-learning-based method for informed molecular docking protocol selection_, as available in [arXiv](https://arXiv).

### How-to

First clone the repository and replicate our package environment:

```bash
conda env create -f conda_env.yml
conda activate cnndockbench
```


Download and uncompress auxiliary data:

```bash
cd cnndockbench
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bUz4ZIzwNZgU67gsgOYlb4v1l2CrN5zk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bUz4ZIzwNZgU67gsgOYlb4v1l2CrN5zk" -O data.tar.gz && rm -rf /tmp/cookies.txt
tar -xf data.tar.gz
```

### Run model

The model implementation provided here requires:

- A (possibly unprotonated) protein `.pdb` file.
- Ligand SMILES with an `.smi` file.
- Binding pocket center coordinates (__x, y, z__).

One can simply run the pretrained model using the `prod.py` script:

```bash
python prod.py -pdb protein.pdb -smi ligands.smi -x 1.61 -y -47.56 -z 6.82 -output PATH
```

The script will generate three files: `rmsd_min.csv`, `rmsd_ave.csv` and `n_rmsd.csv` in the `-output` folder or in the current directory if unspecified.


### Replication of the results

The entire code for replicating the results reported in the paper is found in this repository. You can also precompute some data for faster replication of the results. 

- Step 1 (optional): Run `python precompute.py` and `python net_utils.py`, if you have not downloaded and uncompressed `data.tar.gz` as per instructed.
- Step 2: Run `train.py`. Results will be stored in the `results` folder for each split type.
- Step 3: Profit!


Code for training a new production model is also available in the `train_prod.py` script.

```bash
python train_prod.py
```

This will train a new model from scratch - using GPUs if available - and will place it under the `models` subdirectory.

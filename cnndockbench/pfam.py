import os
import pickle
from glob import glob

from tqdm import tqdm

from utils import home

CLASSES_PATH = os.path.join(home(), 'proteinClasses')


def build_pfam_map(pfam_files):
    pdbid_pfam_map = {}

    for pfam_file in tqdm(pfam_files):
        pfam_family = os.path.basename(pfam_file).strip('.list')
        with open(pfam_file, 'r+') as handle:
            pdbids = handle.readlines()
        
        pdbids = [pdbid.strip('\n').lower() for pdbid in pdbids]

        for pdbid in pdbids:
            pdbid_pfam_map[pdbid] = pfam_family

    with open(os.path.join(CLASSES_PATH, 'pdbid_pfam_map.pt'), 'wb') as handle:
        pickle.dump(pdbid_pfam_map, handle)

    return pdbid_pfam_map


if __name__ == "__main__":
    pfam_files = glob(os.path.join(CLASSES_PATH, '*.list'))
    build_pfam_map(pfam_files)

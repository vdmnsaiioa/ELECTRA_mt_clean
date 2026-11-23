#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
from ase import Atoms
from ase.calculators.vasp import VaspChargeDensity
from utils.custom_vasp_loader import CustomVaspChargeDensity
import torch
from tools.atom_tools import valence_electrons
import time
import os
import tempfile
import lz4
import logging

def convert_to_unnormalized_density(orig_density: np.array,
                                    normalized_transported_density: np.array):
    unnormalized_transported_density = normalized_transported_density * orig_density.sum() / 10
    return unnormalized_transported_density

def normalize_density(n_elec: int,
                          density: torch.tensor,
                          qm9_density: torch.tensor,
                          grid_dict: dict,
                          sys: Atoms,
                         points: torch.tensor,
                        mol_volume: float = None):
    if mol_volume is not None:
        dv = mol_volume / (grid_dict['nx'] * grid_dict['ny'] * grid_dict['nz'])
    else:
        dv = sys.get_volume() / (grid_dict['nx'] * grid_dict['ny'] * grid_dict['nz'])
    if points is None:
        n_elec = n_elec
    else:
        n_elec = qm9_density.flatten()[points].sum() * dv
    sum_target = qm9_density.flatten().sum()
    sum_pred = density.flatten().sum()
    eps = 1e-12
    if sum_pred < eps:
        logging.warning("Predicted density sum below epsilon; skipping normalization.")
        print('Predicted density sum below epsilon; skipping normalization.')
        factor = 1.0
        cd_sum_to_num_electrons = density
    else:
        factor = sum_target / (sum_pred + eps)
        cd_sum_to_num_electrons = density * factor
    #integrated_num_electrons = density.sum() * dv
    #factor = n_elec / integrated_num_electrons
    #cd_sum_to_num_electrons = density * factor
    return cd_sum_to_num_electrons, n_elec, factor


def get_electron_normalized_density(sys: Atoms,
                                    gridrefinement: int,
                                    cd: np.array):
    # This function returns a charge density that actually sums to the required number of electrons by just summing naively (with no integration)
    # Probably not required for anything.
    dv = sys.get_volume() / sys.calc.get_number_of_grid_points().prod()
    integrated_num_electrons = cd.sum() / gridrefinement ** 3
    cd_sum_to_num_electrons = cd / cd.sum() * integrated_num_electrons
    return cd_sum_to_num_electrons


def check_number_of_electrons(sys: Atoms,
                              cd: np.array,
                              gridrefinement: int = 2,
                              grid_dict: dict = None):
    # Get integrated values

    if grid_dict:
        dv = sys.get_volume() / (grid_dict['nx'] * grid_dict['ny'] * grid_dict['nz'])
        integrated_num_electrons = cd.sum() * dv
    else:
        dv = sys.get_volume() / (grid_dict['nx'] * grid_dict['ny'] * grid_dict['nz'])
        integrated_num_electrons = cd.sum() * dv / gridrefinement ** 3
    # print(f"Number of electrons in {sys.get_chemical_formula()} is {np.round(integrated_num_electrons, 2)}")
    return integrated_num_electrons


# Decompressed bytes read when only the structure is needed. The header is
# < 2 KB for QM9; the rest covers the grid lines that CustomVaspChargeDensity's
# second read_vasp attempt looks at (it decides which files get skipped).
_HEADER_ONLY_BYTES = 64 * 1024


def chgcar_to_cd(chgcar_file, header_only=False):
    """Read an .lz4 CHGCAR and return (density tensor, ase.Atoms).

    The file is parsed in memory, with no temp file on the node's /tmp. The
    density is converted with torch.from_numpy instead of
    torch.FloatTensor(list), which gives the same float32 values about 5x faster.

    header_only=True decompresses only the first ~64 KB (~48 KB read from disk
    instead of ~8 MB) and returns (None, atoms). Use it when the density is not
    needed, e.g. energy-only steps of use_fraction_of_density runs.
    """
    filecontent = read_chgcar_bytes(chgcar_file, header_only=header_only)
    return chgcar_bytes_to_cd(filecontent, chgcar_file, header_only=header_only)


def read_chgcar_bytes(chgcar_file, header_only=False):
    """Decompressed CHGCAR bytes (I/O only, so it can run in a thread pool)."""
    with lz4.frame.open(chgcar_file, mode='rb') as fp:
        return fp.read(_HEADER_ONLY_BYTES) if header_only else fp.read()


def chgcar_bytes_to_cd(filecontent, chgcar_file, header_only=False):
    """Parse bytes from read_chgcar_bytes; returns what chgcar_to_cd returns."""
    text = filecontent.decode(errors="ignore") if header_only else filecontent.decode()
    vcd = CustomVaspChargeDensity(None)
    vcd.read_text(text, header_only=header_only)
    charge_density = vcd.chg
    try:
        atoms = vcd.atoms[0]
    except:
        print(f"No atoms in chgcar file {chgcar_file}")
    if header_only:
        return None, atoms
    cd = torch.from_numpy(np.asarray(charge_density, dtype=np.float32)).squeeze(axis=0)
    return cd, atoms

def cd_to_chgcar_mat(original_file: str, atoms: Atoms, cd: np.array, filename: str):
    with lz4.frame.open(original_file, mode='rb') as fp:
        filecontent = fp.read()
    tmpfd, tmppath = tempfile.mkstemp(prefix="tmpchgcar")
    with os.fdopen(tmpfd, "wb") as tmpfile:
        tmpfile.write(filecontent)
    try:
        vcd = CustomVaspChargeDensity(tmppath)
    finally:
        os.remove(tmppath)  # always clean up, even if parsing fails
    vcd.chg = [cd]
    vcd.atoms = [atoms]
    vcd.write(filename, format='chgcar')
    return cd, atoms

def cd_to_chgcar_mol(atoms: Atoms, cd: np.array, filename: str):
    #with open(filename, 'w') as fp:
    #    pass
    #vcd = CustomVaspChargeDensity(filename)
    vcd = CustomVaspChargeDensity(None)  # skip reading
    vcd.chg = [cd]
    vcd.atoms = [atoms]
    vcd.write(filename, format='chgcar')


def get_qm9_density(qm9_path):
    qm9_density, mol = chgcar_to_cd(qm9_path)
    grid_dict = {"nx": qm9_density.shape[0],
                      "ny": qm9_density.shape[1],
                      "nz": qm9_density.shape[2]}
    n_elec_qm9 = check_number_of_electrons(mol, qm9_density, grid_dict=grid_dict)
    return qm9_density, mol, n_elec_qm9, grid_dict





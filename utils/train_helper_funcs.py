from torch.optim import Adam, AdamW, SGD
import torch
import numpy as np
import random
import os
import csv
import json
import re


def set_optimizer(tens_net, config):
    """Instantiate the optimizer configured for ``tens_net``.

    If ``split_params`` is enabled in the configuration and the network
    exposes ``phi_network`` and/or ``rho_network`` submodules, a dedicated
    parameter group is created so a custom learning rate or weight decay can be
    configured via ``energy_initial_lr`` and ``energy_weight_decay``.
    Otherwise all parameters share the base optimiser settings.
    """

    optimizer_name = config['optimizer']
    base_lr = config['initial_lr']
    base_weight_decay = config.get('weight_decay', 0.0)

    param_groups = []
    all_params = list(tens_net.parameters())

    special_params = []
    if hasattr(tens_net, 'phi_network') and tens_net.phi_network is not None:
        special_params.extend(list(tens_net.phi_network.parameters()))
    if hasattr(tens_net, 'rho_network') and tens_net.rho_network is not None:
        special_params.extend(list(tens_net.rho_network.parameters()))

    split_params = config.get('split_params', False)

    if split_params and special_params:
        special_param_ids = {id(p) for p in special_params}
        other_params = [p for p in all_params if id(p) not in special_param_ids]
        print('cucusor')
    else:
        other_params = all_params
        special_params = []

    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': base_weight_decay,
        })

    if special_params:
        energy_lr = config.get('energy_initial_lr', base_lr)
        energy_weight_decay = config.get('energy_weight_decay', base_weight_decay)
        param_groups.append({
            'params': special_params,
            'lr': energy_lr,
            'weight_decay': energy_weight_decay,
        })

    optimizer_kwargs = {}
    if optimizer_name == 'adam':
        optimizer_cls = Adam
        optimizer_kwargs['amsgrad'] = True
    elif optimizer_name == 'adamw':
        optimizer_cls = AdamW
        optimizer_kwargs['amsgrad'] = True
    elif optimizer_name == 'sgd':
        optimizer_cls = SGD
        optimizer_kwargs['momentum'] = config['momentum']
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_name}")

    if not param_groups:
        # Fallback to the full parameter list if no groups were constructed.
        param_groups = all_params

    # Optional fused Adam/AdamW (config['optimizer_fused'], default off).
    # Benchmarked 2026-11-16 on the 723M-param multitask model with torch 2.2.1:
    # fused amsgrad was SLOWER (~211 vs ~74 ms/step) and its updates differ from
    # foreach. Keep it off.
    if config.get('optimizer_fused', False) and optimizer_name in ('adam', 'adamw') \
            and all(p.is_cuda and p.is_floating_point() for p in all_params):
        optimizer_kwargs['fused'] = True

    optimizer = optimizer_cls(param_groups,
                              lr=base_lr,
                              weight_decay=base_weight_decay,
                              **optimizer_kwargs)
    return optimizer


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_all_paths(config, wb_run_name=None):
    pred_dens_path_val = config['pred_dens_path_val']
    pred_dens_path_test = config['pred_dens_path_test']
    density_delta_path_val = config['density_delta_path_val']
    density_delta_path_test = config['density_delta_path_test']
    gaus_pos_path_val = config['gaus_pos_path_val']
    gaus_pos_path_test = config['gaus_pos_path_test']
    if config['niflheim']:
        pred_dens_path_val = config['niflheim_base_path'] + f"split_{config['data_split']}/" + pred_dens_path_val
        pred_dens_path_test = config['niflheim_base_path'] + f"split_{config['data_split']}/" + pred_dens_path_test
        density_delta_path_val = config['niflheim_base_path'] + f"split_{config['data_split']}/" + density_delta_path_val
        density_delta_path_test = config['niflheim_base_path'] + f"split_{config['data_split']}/" + density_delta_path_test
        gaus_pos_path_val = config['niflheim_base_path'] + f"split_{config['data_split']}/" + config['gaus_pos_path_val']
        gaus_pos_path_test = config['niflheim_base_path'] + f"split_{config['data_split']}/" + config['gaus_pos_path_test']
    if config['wandb']:
        pred_dens_path_val = pred_dens_path_val + f"/{wb_run_name}/"
        pred_dens_path_test = pred_dens_path_test + f"/{wb_run_name}/"
        density_delta_path_val = density_delta_path_val + f"/{wb_run_name}/"
        density_delta_path_test = density_delta_path_test + f"/{wb_run_name}/"
        gaus_pos_path_val = gaus_pos_path_val + f"/{wb_run_name}/"
        gaus_pos_path_test = gaus_pos_path_test + f"/{wb_run_name}/"
    config['pred_dens_path_val'] = pred_dens_path_val
    config['pred_dens_path_test'] = pred_dens_path_test
    config['density_delta_path_val'] = density_delta_path_val
    config['density_delta_path_test'] = density_delta_path_test
    config['gaus_pos_path_val'] = gaus_pos_path_val
    config['gaus_pos_path_test'] = gaus_pos_path_test
    os.makedirs(pred_dens_path_val, exist_ok=True)
    os.makedirs(pred_dens_path_test, exist_ok=True)
    os.makedirs(density_delta_path_val, exist_ok=True)
    os.makedirs(density_delta_path_test, exist_ok=True)
    os.makedirs(gaus_pos_path_val, exist_ok=True)
    os.makedirs(gaus_pos_path_test, exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    return config

def get_files(config):
    file_split = config['data_split']
    file_split_path = config['data_split_path'] + f"datasplits_{file_split}.json"
    f = open(file_split_path)
    data = json.load(f)
    train_files_indices = data['train']
    test_files_indices = data['test']
    validation_files_indices = data['validation']
    if file_split != "ECD" and file_split != "ECD_test" and file_split != "nmc":
        train_files = [f"{config['qm9_dens_path']}/{num:06}.CHGCAR.lz4" for num in train_files_indices]
        test_files = [f"{config['qm9_dens_path']}/{num:06}.CHGCAR.lz4" for num in test_files_indices]
        validation_files = [f"{config['qm9_dens_path']}/{num:06}.CHGCAR.lz4" for num in validation_files_indices]
    elif file_split == "ECD" or file_split == "ECD_test":
        train_files = [f"{config['qm9_dens_path']}/{num}.chgcar.lz4" for num in train_files_indices]
        test_files = [f"{config['qm9_dens_path']}/{num}.chgcar.lz4" for num in test_files_indices]
        validation_files = [f"{config['qm9_dens_path']}/{num}.chgcar.lz4" for num in validation_files_indices]
    else:
        train_files = [f"{config['qm9_dens_path']}/{num}.CHGCAR.lz4" for num in train_files_indices]
        test_files = [f"{config['qm9_dens_path']}/{num}.CHGCAR.lz4" for num in test_files_indices]
        validation_files = [f"{config['qm9_dens_path']}/{num}.CHGCAR.lz4" for num in validation_files_indices]
    random.shuffle(train_files)
    random.shuffle(test_files)
    random.shuffle(validation_files)

    # Drop known-unreadable density files (corrupt CHGCARs / missing files).
    # Filtering after the shuffle keeps the order of the remaining files unchanged,
    # so runs resumed from old checkpoints see the same sequence as before.
    bad_files_path = config.get(
        "bad_files_path", os.path.join(config['data_split_path'], "qm9_bad_files.txt"))
    if bad_files_path and os.path.exists(bad_files_path):
        with open(bad_files_path) as fh:
            bad = {os.path.basename(line.strip()) for line in fh if line.strip()}
        n_before = (len(train_files), len(test_files), len(validation_files))
        train_files = [p for p in train_files if os.path.basename(p) not in bad]
        test_files = [p for p in test_files if os.path.basename(p) not in bad]
        validation_files = [p for p in validation_files if os.path.basename(p) not in bad]
        print(f"[get_files] excluded bad files listed in {bad_files_path}: "
              f"train {n_before[0]}->{len(train_files)}, test {n_before[1]}->{len(test_files)}, "
              f"val {n_before[2]}->{len(validation_files)}")

    #train_files = sorted(train_files)
    #test_files = sorted(test_files)
    #validation_files = sorted(validation_files)

    return train_files, test_files, validation_files

def load_csv_to_dict(file_path, key_column, value_column):
    result_dict = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row[key_column]
            value = row[value_column]
            result_dict[key] = value
    return result_dict


def add_run_override_args(parser):
    """Add CLI flags for the per-run config keys that are set from the sbatch script.

    The YAML is only read when a job starts, so keys that change between queued
    runs are passed on the command line instead (the sbatch script is snapshotted
    at submission).
    """
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the random seed.")
    parser.add_argument("--data_split", type=str, default=None,
                        help="Override the data split name.")
    parser.add_argument("--load_model_path", type=str, default=None,
                        help="Override the checkpoint path loaded when load_model is enabled.")

    fraction_group = parser.add_mutually_exclusive_group()
    fraction_group.add_argument("--use_fraction_of_density", dest="use_fraction_of_density", action="store_true",
                                help="Apply the density loss only to indices_for_density (overrides config).")
    fraction_group.add_argument("--no_use_fraction_of_density", dest="use_fraction_of_density", action="store_false",
                                help="Apply the density loss to all training structures (overrides config).")
    parser.set_defaults(use_fraction_of_density=None)

    parser.add_argument("--density_fraction", type=float, default=None,
                        help="Override density_fraction.")
    parser.add_argument("--indices_for_density", type=str, default=None,
                        help="JSON list of indices that get density supervision: a path, or a filename "
                             "in data_splits/density_indices/.")
    parser.add_argument("--lr_id", type=str, default=None,
                        help="Label for the learning-rate schedule; 'normal'/'fine' add lr_normal/lr_fine to the wandb run name.")
    parser.add_argument("--prev_epochs", type=int, default=None,
                        help="Epochs the loaded checkpoint had already trained; adds prevN to the wandb run name.")


def apply_run_overrides(config, args):
    """Apply the overrides added by ``add_run_override_args``."""
    for key in ("seed", "data_split", "load_model_path", "use_fraction_of_density", "density_fraction", "lr_id", "prev_epochs"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    if args.indices_for_density is not None:
        indices_path = args.indices_for_density
        if not os.path.isfile(indices_path):
            # Bare filenames refer to the shared index lists in data_splits/density_indices/
            indices_path = os.path.join(config['data_split_path'], "density_indices", indices_path)
        with open(indices_path) as f:
            config["indices_for_density"] = [int(idx) for idx in json.load(f)]
        config["indices_for_density_file"] = indices_path

    if config.get("use_fraction_of_density", False) and not config.get("indices_for_density"):
        raise ValueError("use_fraction_of_density is enabled but indices_for_density is empty, "
                         "so no structure would get density supervision.")

    return config


def get_run_name_prefix(config, multitask):
    """Build the wandb run-name prefix from the run settings.

    Example: ``mt_mpfull_dens10%_lr_normal_seed101``. The density percentage is the share
    of training structures in ``indices_for_density``, so it reflects what the
    model actually trained on whatever the index file is called.
    """
    # Split names end in the dataset year, e.g. mp10%2025 -> mp10%
    parts = ["mt" if multitask else "eo", re.sub(r"20\d\d$", "", config['data_split'])]

    if multitask:
        if config.get("use_fraction_of_density", False):
            with open(config['data_split_path'] + f"datasplits_{config['data_split']}.json") as f:
                train_indices = set(json.load(f)['train'])
            n_density = len(train_indices & set(config['indices_for_density']))
            pct = 100 * n_density / len(train_indices)
        else:
            pct = 100
        parts.append(f"dens{round(pct, 1):g}%")

    if config.get("lr_id") == "normal":
        parts.append("lr_normal")
    elif config.get("lr_id") == "fine":
        parts.append("lr_fine")

    parts.append(f"seed{config['seed']}")

    if config.get("load_model", False) and config.get("prev_epochs") is not None:
        parts.append(f"prev{config['prev_epochs']}ep")

    return "_".join(parts)

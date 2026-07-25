import ase
import torch


def atom_counts_to_vector(formula: str):
    atom_counts_dict = parse_formula(formula)
    # Define atomic types
    atomic_types = ['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
                    'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
                    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                    'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
                    'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
                    'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
                    'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
                    'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                    'No', 'Lr']

    # Initialize torch tensor with zeros
    atom_vector = torch.zeros(119, dtype=torch.int)

    # Fill in the counts of each atom type
    for atomic_symbol, count in atom_counts_dict.items():
        atomic_index = atomic_types.index(atomic_symbol)
        atom_vector[atomic_index] = count

    return atom_vector


def parse_formula(formula):
    # Parses a molecule into a dictionary of element counts
    elements = {}
    current_element = ""
    current_count = ""
    for char in formula:
        if char.isupper():
            if current_element:
                elements[current_element] = int(current_count) if current_count else 1
            current_element = char
            current_count = ""
        elif char.islower():
            current_element += char
        elif char.isdigit():
            current_count += char
    if current_element:
        elements[current_element] = int(current_count) if current_count else 1
    return elements


def valence_electrons(chemical_formula):
    """
    Returns the total number of valence electrons for a chemical formula,
    reflecting a “minimal” or “standard” approach for most elements, but
    with Ce specifically set to 12 (as observed in your data).
    Pd = 10, Si = 4, etc.

    If any other elements in your data disagree with these defaults,
    update them to match the actual POTCAR ZVAL in your calculations.
    """
    form_dict = parse_formula(chemical_formula)
    total_valence = 0
    for element, count in form_dict.items():
        val_e = valence_dict.get(element, 0)  # or raise an error if missing
        total_valence += val_e * count

    return total_valence

valence_dict = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
        'Na': 7, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
        'K': 9,  'Ca': 10, 'Sc': 11, 'Ti': 12, 'V': 13, 'Cr': 12, 'Mn': 13, 'Fe': 8, 'Co': 9, 'Ni': 16, 'Cu': 17, 'Zn': 12,
        'Ga': 13, 'Ge': 14, 'As': 5, 'Se': 6, 'Br': 7, 'Kr': 8,
        'Rb': 9, 'Sr': 10, 'Y': 11,  'Zr': 10, 'Nb': 13, 'Mo': 14, 'Tc': 13, 'Ru': 14,
        'Rh': 15, 'Pd': 10, 'Ag': 11, 'Cd': 12,
        'In': 13, 'Sn': 4, 'Sb': 5, 'Te': 6, 'I': 7,  'Xe': 8, 'Cs': 9, 'Ba': 10,
        'La': 11, 'Ce': 12, 'Pr': 11, 'Nd': 11, 'Pm': 11, 'Sm': 11, 'Eu': 8, 'Gd': 9,
        'Tb': 9, 'Dy': 9, 'Ho': 9, 'Er': 9, 'Tm': 9, 'Yb': 8, 'Lu': 9,
        'Hf': 10, 'Ta': 11, 'W': 14,  'Re': 13,  'Os': 8, 'Ir': 9, 'Pt': 10,'Au': 11,'Hg': 12,
        'Tl': 13, 'Pb': 14, 'Bi': 15, 'Po': 16, 'At': 7, 'Rn': 8, 'Fr': 1, 'Ra': 2,
        'Ac': 11, 'Th': 12, 'Pa': 13, 'U': 14, 'Np': 15, 'Pu': 16, 'Am': 17, 'Cm': 18,
        'Bk': 3, 'Cf': 3, 'Es': 3, 'Fm': 3, 'Md': 3, 'No': 3, 'Lr': 3,
        'Rf': 4, 'Db': 5, 'Sg': 6, 'Bh': 7, 'Hs': 8, 'Mt': 9,
        'Ds': 10, 'Rg': 11, 'Cn': 12, 'Nh': 3, 'Fl': 4, 'Mc': 5, 'Lv': 6, 'Ts': 7, 'Og': 8,
    }

def main():
    # Example usage
    atom_vector = atom_counts_to_vector("C6H6")
    print(atom_vector)


if __name__ == "__main__":
    main()


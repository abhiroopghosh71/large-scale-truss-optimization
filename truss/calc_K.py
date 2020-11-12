import operator as op
from functools import reduce


def ncr(n, r):
    """Calculates nCr. Taken from https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python"""
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


def calc_K(n_z_nodes_per_side):
    # The design problem gives a truss structure of fixed length with members arranged in a specific way. Within that
    # certain quantities are flexible and can be set by user. To define a unique truss design, we only need the radii
    # of the members and z-coordinates of the bottom portion. Assume the problem specification includes the fact that
    # all top nodes and all bottom nodes have same z-coordinates.
    max_number_of_rules = ncr(260*(n_z_nodes_per_side // 2 + 1) // 10 + n_z_nodes_per_side, 2)
    base_truss = 0 / max_number_of_rules  # Z-coordinates of bottom nodes
    asymm_z = 0.5 * (n_z_nodes_per_side - 1) / max_number_of_rules
    asymm_all = 0.5 * ((n_z_nodes_per_side - 1) + (n_z_nodes_per_side - 1)*10 + n_z_nodes_per_side*4) / max_number_of_rules

    symm = (9  # z
            + (n_z_nodes_per_side // 2) * (3 + 2 + 1) * 2  # straight x. Symmetricity on all sides
            + (n_z_nodes_per_side // 2) * (3 + 2 + 1) * 2  # Straight y and z
            + (3 + 2 + 1)  # Cross
            + (n_z_nodes_per_side // 2) * (3 + 2 + 1)  #  Slant xz
            + (n_z_nodes_per_side // 2) * (3 + 2 + 1) * 2  #  Cross xy
            ) / max_number_of_rules

    # TODO: Recheck with pic help
    symm_z = symm + 0.5 * (n_z_nodes_per_side // 2) / max_number_of_rules
    symm_siv_z = symm + (n_z_nodes_per_side // 2) / max_number_of_rules
    symm_all = symm_z + 0.5 * ((n_z_nodes_per_side // 2 - 1) * (3 + 2 + 1) * 3  # straight x. Symmetricity on all sides
                               + (n_z_nodes_per_side // 2 - 1) * (3 + 2 + 1) * 3  # Straight y and z
                               + (n_z_nodes_per_side // 2 - 1) * (3 + 2 + 1) * 2  # Slant xz
                               + (n_z_nodes_per_side // 2 - 1) * (3 + 2 + 1) * 3) / max_number_of_rules  # Cross xy

    print(f"base_truss = {base_truss}, asymm_z = {asymm_z}, asymm_all = {asymm_all}, symm = {symm}, "
          f"symm_z = {symm_z}, symm_siv_z = {symm_siv_z}, symm_all = {symm_all}")
    return base_truss, asymm_z, asymm_all, symm, symm_z, symm_siv_z, symm_all


if __name__ == '__main__':
    print(ncr(298, 2))
    calc_K(19)

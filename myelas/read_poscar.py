import spglib
import numpy as np
import os
import linecache


class read_poscar(object):
    """
    Read POSCAR-uc 
    --------------------

    INPUT file: POSCAR-uc

    Returns:
    --------
    pos_name       : System name. The POSCAR 1st line.

    lattice_index  : The POSCAR 2nd line.

    lattice        : 3x3 matrix. The POSCAR 3rd-5th line.

    atomname       : The element name. The POSCAR 6th line.

    atomnum        : The number of every element. The POSCAR 7th line.

    postype        : The type of position. The POSCAR 8th line.

    pos            : The position of every atom.

    spg_number     : The space group number of structure.

    """
    def __init__(
        self,
        struct=None,
        pos_name=None,
        lattice_index=None,
        lat=None,
        lat_recell=None,
        atomname=None,
        atomnum=None,
        postype=None,
        pos=None,
        spg_number=None,
    ):

        self.struct = linecache.getlines("POSCAR-uc")
        # read POSCAR to get some paramatrics: sys_name; lattice; atom_name; atom_number; atom_position
        # and get spacegroup_number
        poscar = [line.strip() for line in self.struct]
        num = len(poscar)

        self.pos_name = poscar[0].split()
        self.lat_index = poscar[1].split()
        self.lattice_index = float(self.lat_index[0])

        # matrics of lattice vector

        lat_vector = np.zeros((3, 3))
        index = 0
        for latt in poscar[2:5]:
            latt = latt.split()
            lat_vector[index, :] = latt[0:3]
            index += 1
        self.lattice = lat_vector

        self.atomname = poscar[5].split()
        self.atomnum = poscar[6].split()
        self.postype = poscar[7].split()

        atom_len = len(self.atomname)
        atom_tolnum=0
        for n in np.arange(0,atom_len,1):
            atom_tolnum=int(self.atomnum[n])+atom_tolnum

        # matrics of atom position
        position_vector = np.zeros((atom_tolnum, 3))
        index = 0
        for poss in poscar[8:8+atom_tolnum]:
            poss = poss.split()
            # position_vector[index, 0:3] = poss[0:3]
            position_vector[index, 0] = poss[0]
            position_vector[index, 1] = poss[1]
            position_vector[index, 2] = poss[2]
            index += 1

        self.lat = lat_vector * self.lattice_index
        self.pos = position_vector
        prinumbers = []
        for i in np.arange(0, len(self.atomname)):
            for j in np.arange(0, int(self.atomnum[i]), 1):
                prinumbers.append(i + 1)
        cell = (self.lat, self.pos, prinumbers)
        database = spglib.get_symmetry_dataset(cell, symprec=1e-3)
        self.spg_number = database["number"]

    def system_name(self):
        return self.pos_name

    def latt_index(self):
        return self.lattice_index

    def latti(self):
        return self.lattice

    def atom_name(self):
        return self.atomname

    def atom_number(self):
        return self.atomnum

    def position_type(self):
        return self.postype

    def positions(self):
        return self.pos

    def spacegroup_num(self):
        return self.spg_number

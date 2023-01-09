import os
import numpy as np
from . import read_poscar as readpos
#import read_poscar as readpos
import spglib


class recell(object):
    def __init__(
        self,
        to_pricell = None
    ):
        # read the prigmitive cell
        self.spg_num = readpos.read_poscar().spacegroup_num()
        self.lattindex = readpos.read_poscar().latt_index()
        self.latt = readpos.read_poscar().latti()
        self.atomname = readpos.read_poscar().atom_name()
        self.atomnum = readpos.read_poscar().atom_number()
        self.postype = readpos.read_poscar().position_type()
        self.position = readpos.read_poscar().positions()

        orignumbers = []
        for i in np.arange(0, len(self.atomname)):
            for j in np.arange(0, int(self.atomnum[i]), 1):
                orignumbers.append(i + 1)

        origlattice = self.latt * self.lattindex
        origpositon = self.position

        origcell = (origlattice, origpositon, orignumbers)

        if(to_pricell==False):

        # refine cell
            self.cell_lattice, re_position, re_numbers = spglib.standardize_cell(
                cell=origcell, symprec=1e-3, to_primitive=False
            )
        else:
            self.cell_lattice, re_position, re_numbers = spglib.standardize_cell(
                cell=origcell, symprec=1e-3, to_primitive=True
            )

        re_position_list = list(re_position)
        zipped = list(zip(re_numbers, re_position_list))
        zipsorted = sorted(zipped, key=lambda x: (x[0]))

        re_numbers_sort, re_positon_sort = zip(*zipsorted)

        self.cell_position = np.array(re_positon_sort)

        self.cell_atomnum = []
        for i in np.arange(0, len(self.atomnum), 1):
            num = int(self.atomnum[i]) * (len(re_numbers_sort) / len(orignumbers))
            self.cell_atomnum.append(num)

        # write the refine cell

        writepos = open("RE-POSCAR", mode="w")
        print("recell_poscar", file=writepos)
        print("1.0", file=writepos)

        for m in np.arange(0, 3, 1):
            print(
                format(self.cell_lattice[m, 0], ".10f"),
                "   ",
                format(self.cell_lattice[m, 1], ".10f"),
                "   ",
                format(self.cell_lattice[m, 2], ".10f"),
                file=writepos,
            )

        for j in np.arange(0, len(self.atomname), 1):
            print(self.atomname[j], file=writepos, end=" ")
        print(end="\n", file=writepos)
        for l in np.arange(0, len(self.atomname), 1):
            print(int(self.cell_atomnum[l]), end=" ", file=writepos)
        print(end="\n", file=writepos)
        print(self.postype[0], file=writepos)

        for n in np.arange(0, self.cell_position.shape[0], 1):
            print(
                format(self.cell_position[n, 0], ".10f"),
                "   ",
                format(self.cell_position[n, 1], ".10f"),
                "   ",
                format(self.cell_position[n, 2], ".10f"),
                file=writepos,
            )
        writepos.close()

    def latti(self):
        return self.cell_lattice

    def atom_number(self):
        return self.cell_atomnum

    def positions(self):
        return self.cell_position



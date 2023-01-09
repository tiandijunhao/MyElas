import numpy as np
import linecache
import math
import os

from . import read_poscar as readpos
from . import standard_cell


class phononToElas(object):
    def __init__(self):
        print("Read phonon dispersion to calculate elastic constants.")
        if os.path.exists("input_direct"):
            print("input_direct.out already exists")
        else:
            print("input_direct.out don't exists")

        self.spg_num = readpos.read_poscar().spacegroup_num()
        self.lattindex = readpos.read_poscar().latt_index()
        self.latt = standard_cell.recell(to_pricell=True).latti()
        atom_num = standard_cell.recell(to_pricell=True).atom_number()

        recipvect = np.array([[0.0, 0.0, 0.0]])
        recipvect[0, 0] = (
            self.latt[1, 1] * self.latt[2, 2] - self.latt[2, 1] * self.latt[1, 2]
        )
        recipvect[0, 1] = (
            self.latt[1, 2] * self.latt[2, 0] - self.latt[2, 2] * self.latt[1, 0]
        )
        recipvect[0, 2] = (
            self.latt[1, 0] * self.latt[2, 1] - self.latt[2, 0] * self.latt[1, 1]
        )

        self.volume = (1.0 ** 3.0) * (
            self.latt[0, 0] * recipvect[0, 0]
            + self.latt[0, 1] * recipvect[0, 1]
            + self.latt[0, 2] * recipvect[0, 2]
        )
        potcar = open("POTCAR", mode="r")

        pomass = []
        lines = potcar.readlines()
        for line in lines:
            if "POMASS" in line:
                line_re = line.replace("; ", " ")
                a = line_re.split()
                pomass.append(a[2])

        self.mass = 0
        self.num = 0
        for i in np.arange(0, len(atom_num), 1):
            self.mass = self.mass + float(atom_num[i]) * float(pomass[i])
            self.num = self.num + atom_num[i]
        print("Total mass: ", self.mass)
        self.density = 10000 * float(self.mass) / (6.02 * self.volume)

        with open("input_direct") as kpo:
            kpoinfo = kpo.readlines()
            num_lines = len(kpoinfo)

            self.kpoint = np.zeros(3)
            self.direc = np.zeros(3)
            for i in range(num_lines):
                if self.spg_num >= 16 and self.spg_num <= 74:
                    if "[100]" in kpoinfo[i]:
                        informat_100 = kpoinfo[i].split(" ")
                        self.kpoint[0] = float(informat_100[1])
                        self.direc[0] = float(informat_100[2])
                    elif "[010]" in kpoinfo[i]:
                        informat_010 = kpoinfo[i].split(" ")
                        self.kpoint[1] = float(informat_010[1])
                        self.direc[1] = float(informat_010[2])
                    elif "[001]" in kpoinfo[i]:
                        informat_001 = kpoinfo[i].split(" ")
                        self.kpoint[2] = float(informat_001[1])
                        self.direc[2] = float(informat_001[2])

                if self.spg_num >= 75 and self.spg_num <= 142:
                    if "[100]" in kpoinfo[i]:
                        informat_100 = kpoinfo[i].split(" ")
                        self.kpoint[0] = float(informat_100[1])
                        self.direc[0] = float(informat_100[2])
                    elif "[001]" in kpoinfo[i]:
                        informat_010 = kpoinfo[i].split(" ")
                        self.kpoint[1] = float(informat_010[1])
                        self.direc[1] = float(informat_010[2])
                    elif "[110]" in kpoinfo[i]:
                        informat_001 = kpoinfo[i].split(" ")
                        self.kpoint[2] = float(informat_001[1])
                        self.direc[2] = float(informat_001[2])

                if self.spg_num >= 143 and self.spg_num <= 194:
                    if "[100]" in kpoinfo[i]:
                        informat_100 = kpoinfo[i].split(" ")
                        self.kpoint[0] = float(informat_100[1])
                        self.direc[0] = float(informat_100[2])
                    elif "[001]" in kpoinfo[i]:
                        informat_010 = kpoinfo[i].split(" ")
                        self.kpoint[1] = float(informat_010[1])
                        self.direc[1] = float(informat_010[2])

                if self.spg_num >= 195 and self.spg_num <= 230:
                    if "[100]" in kpoinfo[i]:
                        informat_100 = kpoinfo[i].split(" ")
                        self.kpoint[0] = float(informat_100[1])
                        self.direc[0] = float(informat_100[2])
                    elif "[110]" in kpoinfo[i]:
                        informat_010 = kpoinfo[i].split(" ")
                        self.kpoint[1] = float(informat_010[1])
                        self.direc[1] = float(informat_010[2])
                    elif "[111]" in kpoinfo[i]:
                        informat_001 = kpoinfo[i].split(" ")
                        self.kpoint[2] = float(informat_001[1])
                        self.direc[2] = float(informat_001[2])

    def cal_deriv(self, x, y):
        diff_x = []

        for i, j in zip(x[0::], x[1::]):
            diff_x.append(j - i)

        diff_y = []
        for i, j in zip(y[0::], y[1::]):
            diff_y.append(j - i)

        slopes = []
        for i in range(len(diff_y)):
            slopes.append(diff_y[i] / diff_x[i])

        deriv = []
        for i, j in zip(slopes[0::], slopes[1::]):
            deriv.append((0.5 * (i + j)))
        deriv.insert(0, slopes[0])
        deriv.append(slopes[-1])

        return deriv

    def calc_elas_phonopy(self, phonon_file=None):
        modes = np.zeros((3, 3))

        for j in np.arange(0, 3, 1):
            y_0 = np.zeros((1, 3))
            y_1 = np.zeros((1, 3))

            with open(phonon_file) as drec:
                text = drec.readlines()
                num_lines = len(text)
                num = 0
                for i in range(num_lines):
                    if "natom" in text[i]:
                        num_atom = int(text[i].split(" ")[1])

                    if "distance" in text[i]:
                        dis = text[i].split("   ")
                        a = dis[1]
                        # print(a)
                        if num == 0:
                            if self.kpoint[j] == float(a):
                                num = num + 1
                                if self.direc[j] == 1:
                                    a1 = text[
                                        i + 4 + num_atom * 3 * 2
                                    ].split("   ")[1]
                                    if float(a1) == self.kpoint[j]:
                                        x = text[
                                            i
                                            + 2
                                            * (4 + num_atom * 3 * 2)
                                        ].split("   ")[1]
                                        n = i + 2 * (
                                            4 + num_atom * 3 * 2)
            
                                    else:
                                        x = text[
                                            i + 4 + num_atom * 3 * 2
                                        ].split("   ")[1]
                                        n = i + (4 + num_atom * 3 * 2)
                                    diff_x = abs(float(x) - float(a))
                                    y_0[0, 0] = text[i + 3].split()[1]
                                    y_0[0, 1] = text[i + 5].split()[1]
                                    y_0[0, 2] = text[i + 7].split()[1]

                                    y_1[0, 0] = text[n + 3].split()[1]
                                    y_1[0, 1] = text[n + 5].split()[1]
                                    y_1[0, 2] = text[n + 7].split()[1]

                                    diff_y = abs(y_1 - y_0)
                                    #diff_y1 = sorted(diff_y[0, :])
                                    diff_xy = diff_y / diff_x

                                    modes[j, :] = diff_xy * 100

                                elif self.direc[j] == -1:
                                    a1 = text[
                                        i - (4 + num_atom * 3 * 2)
                                    ].split("   ")[1]
                                    if float(a1) == self.kpoint[j]:
                                        x = text[
                                            i
                                            - 2
                                            * ((4 + num_atom * 3 * 2))
                                        ].split("   ")[1]
                                        n = i - 2 * (
                                            (4 + num_atom * 3 * 2)
                                        )
                                    else:
                                        x = text[
                                            i - (4 + num_atom * 3 * 2)
                                        ].split("   ")[1]
                                        n = i - (4 + num_atom * 3 * 2)

                                    # print(x)
                                    diff_x = abs(float(x) - float(a))
                                    y_0[0, 0] = text[i + 3].split()[1]
                                    y_0[0, 1] = text[i + 5].split()[1]
                                    y_0[0, 2] = text[i + 7].split()[1]

                                    y_1[0, 0] = text[n + 3].split()[1]
                                    y_1[0, 1] = text[n + 5].split()[1]
                                    y_1[0, 2] = text[n + 7].split()[1]

                                    diff_y = abs(y_1 - y_0)
                                    #diff_y1 = sorted(diff_y[0, :])
                                    diff_xy = diff_y / diff_x

                                    modes[j, :] = diff_xy * 100

        C_tensor = (modes ** 2) * self.density * 10 ** (-9)

        self.phonon_elastic(modes, C_tensor)

    def calc_elas_phonopy_old(self, phonon_file=None):
        modes = np.zeros((3, 3))

        for j in np.arange(0, 3, 1):
            y_0 = np.zeros((1, 3))
            y_1 = np.zeros((1, 3))

            with open(phonon_file) as drec:
                text = drec.readlines()
                num_lines = len(text)
                num = 0
                for i in range(num_lines):
                    if "natom" in text[i]:
                        num_atom = int(text[i].split(" ")[1])

                    if "distance" in text[i]:
                        dis = text[i].split("   ")
                        a = dis[1]
                        # print(a)
                        if num == 0:
                            if self.kpoint[j] == float(a):
                                num = num + 1
                                if self.direc[j] == 1:
                                    a1 = text[
                                        i + 4 + num_atom * 3 * (3 + num_atom * 4)
                                    ].split("   ")[1]
                                    if float(a1) == self.kpoint[j]:
                                        x = text[
                                            i
                                            + 2
                                            * (4 + num_atom * 3 * (3 + num_atom * 4))
                                        ].split("   ")[1]
                                        n = i + 2 * (
                                            4 + num_atom * 3 * (3 + num_atom * 4)
                                        )
                                    else:
                                        x = text[
                                            i + 4 + num_atom * 3 * (3 + num_atom * 4)
                                        ].split("   ")[1]
                                        n = i + (4 + num_atom * 3 * (3 + num_atom * 4))
                                    diff_x = abs(float(x) - float(a))
                                    y_0[0, 0] = text[i + 3].split()[1]
                                    y_0[0, 1] = text[i + 6 + num_atom * 4].split()[1]
                                    y_0[0, 2] = text[i + 9 + 2 * num_atom * 4].split()[
                                        1
                                    ]

                                    y_1[0, 0] = text[n + 3].split()[1]
                                    y_1[0, 1] = text[n + 6 + num_atom * 4].split()[1]
                                    y_1[0, 2] = text[n + 9 + 2 * num_atom * 4].split()[
                                        1
                                    ]

                                    diff_y = abs(y_1 - y_0)
                                    diff_y1 = sorted(diff_y[0, :])
                                    diff_xy = np.array(diff_y1) / diff_x

                                    modes[j, :] = diff_xy * 100

                                elif self.direc[j] == -1:
                                    a1 = text[
                                        i + 4 + num_atom * 3 * (3 + num_atom * 4)
                                    ].split("   ")[1]
                                    if float(a1) == self.kpoint[j]:
                                        x = text[
                                            i
                                            - 2
                                            * (4 + num_atom * 3 * (3 + num_atom * 4))
                                        ].split("   ")[1]
                                        n = i - 2 * (
                                            4 + num_atom * 3 * (3 + num_atom * 4)
                                        )
                                    else:
                                        x = text[
                                            i - 4 - num_atom * 3 * (3 + num_atom * 4)
                                        ].split("   ")[1]
                                        n = i - (4 + num_atom * 3 * (3 + num_atom * 4))

                                    # print(x)
                                    diff_x = abs(float(x) - float(a))
                                    y_0[0, 0] = text[i + 3].split()[1]
                                    y_0[0, 1] = text[i + 6 + num_atom * 4].split()[1]
                                    y_0[0, 2] = text[i + 9 + 2 * num_atom * 4].split()[
                                        1
                                    ]

                                    # print(y_0)
                                    y_1[0, 0] = text[n + 3].split()[1]
                                    y_1[0, 1] = text[n + 6 + num_atom * 4].split()[1]
                                    y_1[0, 2] = text[n + 9 + 2 * num_atom * 4].split()[
                                        1
                                    ]
                                    sorted(y_1,)
                                    # print(y_1)
                                    diff_y = abs(y_1 - y_0)
                                    diff_y1 = sorted(diff_y[0, :])
                                    diff_xy = np.array(diff_y1) / diff_x

                                    modes[j, :] = diff_xy * 100

        C_tensor = (modes ** 2) * self.density * 10 ** (-9)

        self.phonon_elastic(modes, C_tensor)
    
    
    def calc_elas_alamode(self, phonon_file=None, phonon_type=None):
        modes = np.zeros((3, 3))

        phonon = np.loadtxt(phonon_file)
        num_lines = len(phonon[:, 0])

        for j in np.arange(0, 3, 1):
            y_0 = np.zeros((1, 3))
            y_1 = np.zeros((1, 3))

            num = 0

            for i in np.arange(0, num_lines, 1):
                if num == 0:
                    if phonon_type == "scph":
                        if phonon[i, 1] == self.kpoint[j]:
                            num = num + 1
                            if self.direc[j] == 1:
                                y_0[0, 0] = phonon[i, 2]
                                y_0[0, 1] = phonon[i, 3]
                                y_0[0, 2] = phonon[i, 4]
                                if phonon[i + 1, 1] == self.kpoint[j]:
                                    diff_x = abs(phonon[i + 2, 1] - phonon[i, 1])
                                    y_1[0, 0] = phonon[i + 2, 2]
                                    y_1[0, 1] = phonon[i + 2, 3]
                                    y_1[0, 2] = phonon[i + 2, 4]
                                else:
                                    diff_x = abs(phonon[i + 1, 1] - phonon[i, 1])
                                    y_1[0, 0] = phonon[i + 1, 2]
                                    y_1[0, 1] = phonon[i + 1, 3]
                                    y_1[0, 2] = phonon[i + 1, 4]

                                diff_y = abs(y_1 - y_0)
                                print(diff_x)
                                diff_xy = diff_y / (diff_x / (0.529177208 * 2 * np.pi))

                                modes[j, :] = diff_xy * 100 / 33.35640952

                            if self.direc[j] == -1:
                                y_0[0, 0] = phonon[i, 2]
                                y_0[0, 1] = phonon[i, 3]
                                y_0[0, 2] = phonon[i, 4]
                                if phonon[i - 1, 0] == self.kpoint[j]:
                                    diff_x = abs(phonon[i - 2, 1] - phonon[i, 1])
                                    y_1[0, 0] = phonon[i - 2, 2]
                                    y_1[0, 1] = phonon[i - 2, 3]
                                    y_1[0, 2] = phonon[i - 2, 4]
                                else:
                                    diff_x = abs(phonon[i - 1, 1] - phonon[i, 1])
                                    y_1[0, 0] = phonon[i - 1, 2]
                                    y_1[0, 1] = phonon[i - 1, 3]
                                    y_1[0, 2] = phonon[i - 1, 4]

                                diff_y = abs(y_1 - y_0)

                                diff_xy = diff_y / (diff_x / (0.529177208 * 2 * np.pi))

                                modes[j, :] = diff_xy * 100 / 33.35640952

                    else:
                        if phonon[i, 0] == self.kpoint[j]:
                            num = num + 1
                            if self.direc[j] == 1:
                                y_0[0, 0] = phonon[i, 1]
                                y_0[0, 1] = phonon[i, 2]
                                y_0[0, 2] = phonon[i, 3]
                                if phonon[i + 1, 0] == self.kpoint[j]:
                                    diff_x = abs(phonon[i + 2, 0] - phonon[i, 0])
                                    y_1[0, 0] = phonon[i + 2, 1]
                                    y_1[0, 1] = phonon[i + 2, 2]
                                    y_1[0, 2] = phonon[i + 2, 3]
                                else:
                                    diff_x = abs(phonon[i + 1, 0] - phonon[i, 0])
                                    y_1[0, 0] = phonon[i + 1, 1]
                                    y_1[0, 1] = phonon[i + 1, 2]
                                    y_1[0, 2] = phonon[i + 1, 3]

                                diff_y = abs(y_1 - y_0)

                                diff_xy = diff_y / (diff_x / (0.529177208 * 2 * np.pi))

                                modes[j, :] = diff_xy * 100 / 33.35640952

                            if self.direc[j] == -1:
                                y_0[0, 0] = phonon[i, 1]
                                y_0[0, 1] = phonon[i, 2]
                                y_0[0, 2] = phonon[i, 3]
                                if phonon[i - 1, 0] == self.kpoint[j]:
                                    diff_x = abs(phonon[i - 2, 0] - phonon[i, 0])
                                    y_1[0, 0] = phonon[i - 2, 1]
                                    y_1[0, 1] = phonon[i - 2, 2]
                                    y_1[0, 2] = phonon[i - 2, 3]
                                else:
                                    diff_x = abs(phonon[i - 1, 0] - phonon[i, 0])
                                    y_1[0, 0] = phonon[i - 1, 1]
                                    y_1[0, 1] = phonon[i - 1, 2]
                                    y_1[0, 2] = phonon[i - 1, 3]

                                diff_y = abs(y_1 - y_0)

                                diff_xy = diff_y / (diff_x / (0.529177208 * 2 * np.pi))

                                modes[j, :] = diff_xy * 100 / 33.35640952

        C_tensor = (modes ** 2) * self.density * 10 ** (-9)
        self.phonon_elastic(modes, C_tensor)

    def calc_elas_tdep(self, phonon_file=None):
        modes = np.zeros((3, 3))

        phonon = np.loadtxt(phonon_file)
        num_lines = len(phonon[:, 0])

        for j in np.arange(0, 3, 1):
            y_0 = np.zeros((1, 3))
            y_1 = np.zeros((1, 3))

            num = 0

            for i in np.arange(0, num_lines, 1):
                if num == 0:
                    if abs(phonon[i, 0] - self.kpoint[j]) < 5 * 10 ** (-6):
                        num = num + 1
                        if self.direc[j] == 1:
                            y_0[0, 0] = phonon[i, 1]
                            y_0[0, 1] = phonon[i, 2]
                            y_0[0, 2] = phonon[i, 3]
                            if abs(phonon[i + 1, 0] - self.kpoint[j]) < 5 * 10 ** (-6):
                                diff_x = abs(phonon[i + 2, 0] - phonon[i, 0])
                                y_1[0, 0] = phonon[i + 2, 1]
                                y_1[0, 1] = phonon[i + 2, 2]
                                y_1[0, 2] = phonon[i + 2, 3]
                            else:
                                diff_x = abs(phonon[i + 1, 0] - phonon[i, 0])
                                y_1[0, 0] = phonon[i + 1, 1]
                                y_1[0, 1] = phonon[i + 1, 2]
                                y_1[0, 2] = phonon[i + 1, 3]

                            diff_y = abs(y_1 - y_0)

                            diff_xy = diff_y / diff_x

                            modes[j, :] = diff_xy * 100 / (2)

                        if self.direc[j] == -1:
                            y_0[0, 0] = phonon[i, 1]
                            y_0[0, 1] = phonon[i, 2]
                            y_0[0, 2] = phonon[i, 3]
                            if abs(phonon[i - 1, 0] - self.kpoint[j]) < 5 * 10 ** (-6):
                                diff_x = abs(phonon[i - 2, 0] - phonon[i, 0])
                                y_1[0, 0] = phonon[i - 2, 1]
                                y_1[0, 1] = phonon[i - 2, 2]
                                y_1[0, 2] = phonon[i - 2, 3]
                            else:
                                diff_x = abs(phonon[i - 1, 0] - phonon[i, 0])
                                y_1[0, 0] = phonon[i - 1, 1]
                                y_1[0, 1] = phonon[i - 1, 2]
                                y_1[0, 2] = phonon[i - 1, 3]

                            diff_y = abs(y_1 - y_0)

                            diff_y = abs(y_1 - y_0)

                            diff_xy = diff_y / diff_x

                            modes[j, :] = diff_xy * 100 / (2)

        C_tensor = (modes ** 2) * self.density * 10 ** (-9)
        self.phonon_elastic(modes, C_tensor)

    def phonon_elastic(self, modes=None, C_tensor=None):

        phonon_elas = open("phonon_elastic.out", mode="w")

        if self.spg_num >= 16 and self.spg_num <= 74:

            C11 = C_tensor[0, 2]
            C22 = C_tensor[1, 2]
            C33 = C_tensor[2, 2]
            C44 = (C_tensor[1, 1] + C_tensor[2, 1]) / 2.0
            C55 = (C_tensor[0, 1] + C_tensor[2, 0]) / 2.0
            C66 = (C_tensor[0, 0] + C_tensor[1, 0]) / 2.0

            print("Orthorhombic crystal:", file=phonon_elas)
            print("\n", end="", file=phonon_elas)

            print("C11  C22  C33  C44  C55  C66 (GPa)", file=phonon_elas)
            print(
                "{:.3f}   {:.3f}   {:.3f}   {:.3f}   {:.3f}   {:.3f}".format(
                    C11, C22, C33, C44, C55, C66
                ),
                file=phonon_elas,
            )
            print("\n", end="", file=phonon_elas)

            print("Single-crystalline sound velocity (m/s)", file=phonon_elas)
            print(
                "100  vl = {:.3f};  vs1 = {:.3f}; vs2 = {:.3f}".format(
                    modes[0, 2], modes[0, 0], modes[0, 1]
                ),
                file=phonon_elas,
            )
            print(
                "010  vl = {:.3f};  vs1 = {:.3f}; vs2 = {:.3f}".format(
                    modes[1, 2], modes[1, 0], modes[1, 1]
                ),
                file=phonon_elas,
            )
            print(
                "001  vl = {:.3f};  vs1 = {:.3f}; vs2 = {:.3f}".format(
                    modes[2, 2], modes[2, 0], modes[2, 1]
                ),
                file=phonon_elas,
            )
            print("\n", end="", file=phonon_elas)

            print(
                "unit cell volume :  {:.4f} A^3".format(self.volume), file=phonon_elas
            )
            print(
                "unit cell density:  {:.4f} kg/m^3".format(self.density),
                file=phonon_elas,
            )

        elif self.spg_num >= 75 and self.spg_num <= 142:
            C11 = C_tensor[0, 2]
            C33 = C_tensor[1, 2]
            C44 = C_tensor[0, 0]
            C66 = (C_tensor[0, 1] + C_tensor[2, 0] + C_tensor[1, 0]) / 3
            # c1 = (2 * C_tensor[2, 1] + 2 * C_tensor[2, 2] - 2 * C_tensor[2, 0]) / 2
            C12 = C11 - 2 * C_tensor[2, 0]

            print("Tetragonal crystal:", file=phonon_elas)
            print("\n", end="", file=phonon_elas)

            print("C11  C33  C44  C66  C12 (GPa)", file=phonon_elas)
            print(
                "{:.3f}   {:.3f}   {:.3f}   {:.3f}   {:.3f}".format(
                    C11, C33, C44, C66, C12
                ),
                file=phonon_elas,
            )
            print("\n", end="", file=phonon_elas)
            print("Single-crystalline sound velocity (m/s)", file=phonon_elas)
            print(
                "100  vl = {:.3f};  vs1 = {:.3f}; vs2 = {:.3f}".format(
                    modes[0, 2], modes[0, 0], modes[0, 1]
                ),
                file=phonon_elas,
            )
            print(
                "001  vl = {:.3f};  vs1 = {:.3f}; vs2 = {:.3f}".format(
                    modes[1, 2], modes[1, 0], modes[1, 1]
                ),
                file=phonon_elas,
            )
            print(
                "110  vl = {:.3f};  vs1 = {:.3f}; vs2 = {:.3f}".format(
                    modes[2, 2], modes[2, 0], modes[2, 1]
                ),
                file=phonon_elas,
            )
            print("\n", end="", file=phonon_elas)

            print(
                "unit cell volume :  {:.4f} A^3".format(self.volume), file=phonon_elas
            )
            print(
                "unit cell density:  {:.4f} kg/m^3".format(self.density),
                file=phonon_elas,
            )

        elif self.spg_num >= 143 and self.spg_num <= 194:
            C11 = C_tensor[0, 0]
            C12 = C_tensor[0, 0] - 2 * C_tensor[0, 2]
            C33 = C_tensor[1, 2]
            C44 = (C_tensor[1, 0] + C_tensor[1, 1] + C_tensor[0, 1]) / 3.0

            print("Hexaginoal and Triginol crystal:", file=phonon_elas)
            print("\n", end="", file=phonon_elas)

            print("C11  C33  C44  C12 (GPa)", file=phonon_elas)
            print(
                "{:.3f}   {:.3f}   {:.3f}   {:.3f}".format(C11, C33, C44, C12),
                file=phonon_elas,
            )
            print("\n", end="", file=phonon_elas)
            print("Single-crystalline sound velocity (m/s)", file=phonon_elas)
            print(
                "100  vl = {:.3f};  vs1 = {:.3f}; vs2 = {:.3f}".format(
                    modes[0, 2], modes[0, 0], modes[0, 1]
                ),
                file=phonon_elas,
            )
            print(
                "001  vl = {:.3f};  vs1 = {:.3f}; vs2 = {:.3f}".format(
                    modes[1, 2], modes[1, 0], modes[1, 1]
                ),
                file=phonon_elas,
            )
            print("\n", end="", file=phonon_elas)

            print(
                "unit cell volume :  {:.4f} A^3".format(self.volume), file=phonon_elas
            )
            print(
                "unit cell density:  {:.4f} kg/m^3".format(self.density),
                file=phonon_elas,
            )

        elif self.spg_num >= 195 and self.spg_num <= 230:
            C11 = C_tensor[0, 2]
            C12 = C_tensor[0, 2] - 2 * C_tensor[1, 0]
            C44 = (C_tensor[0, 0] + C_tensor[0, 1]) / 2.0

            print("Cubic crystal:", file=phonon_elas)
            print("\n", end="", file=phonon_elas)

            print("C11  C12  C44 (GPa)", file=phonon_elas)
            print(
                "{:.3f}   {:.3f}   {:.3f}".format(C11, C12, C44), file=phonon_elas,
            )
            print("\n", end="", file=phonon_elas)
            print("Single-crystalline sound velocity (m/s)", file=phonon_elas)
            print(
                "100  vl = {:.3f};  vs1 = {:.3f}; vs2 = {:.3f}".format(
                    modes[0, 2], modes[0, 0], modes[0, 1]
                ),
                file=phonon_elas,
            )
            print(
                "110  vl = {:.3f};  vs1 = {:.3f}; vs2 = {:.3f}".format(
                    modes[1, 2], modes[1, 0], modes[1, 1]
                ),
                file=phonon_elas,
            )
            print(
                "111  vl = {:.3f};  vs1 = {:.3f}; vs2 = {:.3f}".format(
                    modes[2, 2], modes[2, 0], modes[2, 1]
                ),
                file=phonon_elas,
            )
            print("\n", end="", file=phonon_elas)

            print(
                "unit cell volume :  {:.4f} A^3".format(self.volume), file=phonon_elas
            )
            print(
                "unit cell density:  {:.4f} kg/m^3".format(self.density),
                file=phonon_elas,
            )


# phononToElas().calc_elas_alamode(phonon_file="U_300.out",phonon_type="harmonic")


import os
import numpy as np
from . import read_poscar as readpos
from . import standard_cell
from . import solve_elastic_3D
from . import plot_fit_E
from . import strain_matrix_string

class solve_elas3D(object):
    def __init__(self):
        print("Solve 3rd elastic constants of bulk materials.")
        self.spg_num = readpos.read_poscar().spacegroup_num()
        self.lattindex = readpos.read_poscar().latt_index()
        self.latt = standard_cell.recell(to_pricell=False).latti()
        atom_num = standard_cell.recell(to_pricell=False).atom_number()

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

    def solve(
        self, strain_max=None, strain_num=None,
    ):
        """
        To solve the second elastics
        strain_max : must be equal to the maxinum of generating strain poscar.
        strain_num : must be equal to the number of strain poscar in one elastic independent.
        """

        # if self.spg_num >= 1 and self.spg_num <= 2:
        #    nelastic = 21
        #
        # elif self.spg_num >= 3 and self.spg_num <= 15:
        #    nelastic = 13

        if self.spg_num >= 16 and self.spg_num <= 74:
            nelastic = 20

        elif self.spg_num >= 75 and self.spg_num <= 142:
            nelastic = 12

        elif self.spg_num >= 149 and self.spg_num <= 167:
            nelastic = 14

        elif self.spg_num >= 168 and self.spg_num <= 176:
            nelastic = 12

        elif self.spg_num >= 177 and self.spg_num <= 194:
            nelastic = 10

        elif self.spg_num >= 195 and self.spg_num <= 206:
            nelastic = 8

        elif self.spg_num >= 207 and self.spg_num <= 230:
            nelastic = 6

        strain_matrix_str=strain_matrix_string.Elastics_3D(nelastic=nelastic)
        
        starin_step = 2.0 * strain_max / (strain_num - 1)
        strain_param = np.arange(-strain_max, strain_max + 0.0001, starin_step)

        strain_energy = {}
        fit_coeffs_3rd = []
        fit_coeffs_2nd = []

        E_strain_data = open("E_Strain.out", mode="w")
        print("Space group {:0>3d}".format(self.spg_num), file=E_strain_data)
        for nelas in np.arange(0, nelastic, 1):
            print(
                "#nelastic {:0>2d}  strain  energy  (E-E0)/V0".format(nelas + 1),
                file=E_strain_data,
            )
            energy = []
            fit_energy = []
            for ndef in np.arange(0, strain_num, 1):
                full_path = (
                    "thirdnelastic_"
                    + str(format(nelas + 1, "02d"))
                    + "/strain_"
                    + str(format(ndef + 1, "03d"))
                )

                # readout = open(
                #    "OSZICAR_"
                #    + str(format(nelas + 1, "02d"))
                #    + "_"
                #    + str(format(ndef + 1, "03d")),
                #    mode="r",
                # )
                readout = open(full_path + "/OUTCAR", mode="r")

                outlines = readout.readlines()

                for i in np.arange(0, len(outlines), 1):
                    if "ISMEAR" in outlines[i]:
                        et = outlines[i].split()[2]
                    if "energy  without entropy=" in outlines[i]:
                        if "enthalpy is  TOTEN" in outlines[i + 1]:
                            E = outlines[i + 1].split()
                            energy.append(float(E[4]))
                        else:
                            if et == "-1;":
                                E = outlines[i - 2].split()
                                energy.append(float(E[4]))
                            else:
                                E = outlines[i].split()
                                energy.append(float(E[6]))

            strain_energy[nelas] = np.array(energy)

            for j in np.arange(0, strain_num, 1):

                fit_param = (np.array(energy)[j] - np.array(energy).min()) / (
                    self.volume
                )
                fit_energy.append(fit_param)
                print(
                    "{:.6f}  {:.8f}  {:.8f}".format(
                        strain_param[j], energy[j], fit_param
                    ),
                    file=E_strain_data,
                )

            fit_poly = np.polyfit(strain_param, np.array(fit_energy), 4)
            fit_func = np.poly1d(fit_poly)
            coffs = fit_func.coeffs * 160.21766208
            fit_coeffs_3rd.append(coffs[1])
            fit_coeffs_2nd.append(coffs[2])

            # plot fit function
            plot_fit_E.plot_energy_fit().plot_energy(
                x=strain_param,
                y=fit_energy,
                fit_func=fit_func,
                figname="Nelastic {:0>1d}".format(nelas + 1),
                strain_str = strain_matrix_str[nelas]
            )

        if self.spg_num >= 16 and self.spg_num <= 74:
            self.__orthorhombic_solve(
                c_coeffs_3rd=fit_coeffs_3rd, c_coeffs_2nd=fit_coeffs_2nd
            )

        elif self.spg_num >= 75 and self.spg_num <= 142:
            self.tetragonal_solve(
                c_coeffs_3rd=fit_coeffs_3rd, c_coeffs_2nd=fit_coeffs_2nd
            )

        # elif self.spg_num >= 143 and self.spg_num <= 148:
        #    self.__rhombohedral_II_solve(c_coeffs_3rd=fit_coeffs_3rd, c_coeffs_2nd=fit_coeffs_2nd)

        elif self.spg_num >= 149 and self.spg_num <= 167:
            self.rhombohedral3rd_I_solve(
                c_coeffs_3rd=fit_coeffs_3rd, c_coeffs_2nd=fit_coeffs_2nd
            )

        elif self.spg_num >= 168 and self.spg_num <= 176:
            self.__HexaII3rd_solve(
                c_coeffs_3rd=fit_coeffs_3rd, c_coeffs_2nd=fit_coeffs_2nd
            )

        elif self.spg_num >= 177 and self.spg_num <= 194:
            self.__HexaI3rd_solve(
                c_coeffs_3rd=fit_coeffs_3rd, c_coeffs_2nd=fit_coeffs_2nd
            )

        elif self.spg_num >= 195 and self.spg_num <= 206:
            self.__cubicII3rd_solve(
                c_coeffs_3rd=fit_coeffs_3rd, c_coeffs_2nd=fit_coeffs_2nd
            )

        elif self.spg_num >= 207 and self.spg_num <= 230:
            self.__cubicI3rd_solve(
                c_coeffs_3rd=fit_coeffs_3rd, c_coeffs_2nd=fit_coeffs_2nd
            )

    def __cubicI3rd_solve(self, c_coeffs_3rd=None, c_coeffs_2nd=None):

        elas_2nd = np.zeros((6, 6))
        elas_2nd[0, 0] = 2 * (3 * c_coeffs_2nd[1] - c_coeffs_2nd[2]) / 3.0
        elas_2nd[0, 1] = (2 * c_coeffs_2nd[2] - 3 * c_coeffs_2nd[1]) / 3.0
        elas_2nd[5, 5] = c_coeffs_2nd[5] / 6.0

        elas_2nd[1, 1] = elas_2nd[2, 2] = elas_2nd[0, 0]
        elas_2nd[3, 3] = elas_2nd[4, 4] = elas_2nd[5, 5]
        elas_2nd[1, 0] = elas_2nd[0, 2] = elas_2nd[2, 0] = elas_2nd[1, 2] = elas_2nd[
            2, 1
        ] = elas_2nd[0, 1]

        solve_elastic_3D.solve_elas3D().elasproperties(Celas=elas_2nd)

        elas_3rd_111 = 6.0 * c_coeffs_3rd[0]
        elas_3rd_112 = (6 * c_coeffs_3rd[1] - 12 * c_coeffs_3rd[0]) / 6.0
        elas_3rd_123 = c_coeffs_3rd[2] - 3 * c_coeffs_3rd[1] + 3 * c_coeffs_3rd[0]
        elas_3rd_144 = (6 * c_coeffs_3rd[3] - 6 * c_coeffs_3rd[0]) / 12.0
        elas_3rd_166 = (6 * c_coeffs_3rd[4] - 6 * c_coeffs_3rd[0]) / 12.0
        elas_3rd_456 = (6 * c_coeffs_3rd[5]) / 48.0

        elas3rd_file = open("ELADAT_3rd", mode="w")
        print("The third elastic constants.", file=elas3rd_file)
        print(
            "Please check the number of elastic constants for your structure.",
            file=elas3rd_file,
        )
        print("\n", file=elas3rd_file)

        print("C111 = ", format(elas_3rd_111, ".4f"), file=elas3rd_file)
        print("C112 = ", format(elas_3rd_112, ".4f"), file=elas3rd_file)
        print("C123 = ", format(elas_3rd_123, ".4f"), file=elas3rd_file)
        print("\n", file=elas3rd_file)

        print("C144 = ", format(elas_3rd_144, ".4f"), file=elas3rd_file)
        print("C166 = ", format(elas_3rd_166, ".4f"), file=elas3rd_file)
        print("C456 = ", format(elas_3rd_456, ".4f"), file=elas3rd_file)
        
        print("\n", end="", file=elas3rd_file)
        print("Please cite: Comput. Phys. Commun., 281 (2022), 108495", file=elas3rd_file)

    def __cubicII3rd_solve(self, c_coeffs_3rd=None, c_coeffs_2nd=None):

        elas_2nd = np.zeros((6, 6))
        elas_2nd[0, 0] = 2 * (3 * c_coeffs_2nd[1] - c_coeffs_2nd[2]) / 3.0
        elas_2nd[0, 1] = (2 * c_coeffs_2nd[2] - 3 * c_coeffs_2nd[1]) / 3.0
        elas_2nd[5, 5] = c_coeffs_2nd[5] / 6.0

        elas_2nd[1, 1] = elas_2nd[2, 2] = elas_2nd[0, 0]
        elas_2nd[3, 3] = elas_2nd[4, 4] = elas_2nd[5, 5]
        elas_2nd[1, 0] = elas_2nd[0, 2] = elas_2nd[2, 0] = elas_2nd[1, 2] = elas_2nd[
            2, 1
        ] = elas_2nd[0, 1]

        solve_elastic_3D.solve_elas3D().elasproperties(Celas=elas_2nd)

        elas_3rd_111 = 6.0 * c_coeffs_3rd[0]
        elas_3rd_112 = (6 * c_coeffs_3rd[1] - 12 * c_coeffs_3rd[0]) / 6.0
        elas_3rd_113 = (6 * c_coeffs_3rd[2] - 12 * c_coeffs_3rd[0]) / 6.0
        elas_3rd_123 = c_coeffs_3rd[3] - 3 * c_coeffs_3rd[1] + 3 * c_coeffs_3rd[0]
        elas_3rd_144 = (6 * c_coeffs_3rd[4] - 6 * c_coeffs_3rd[0]) / 12.0
        elas_3rd_155 = (6 * c_coeffs_3rd[5] - 6 * c_coeffs_3rd[0]) / 12.0
        elas_3rd_166 = (6 * c_coeffs_3rd[6] - 6 * c_coeffs_3rd[0]) / 12.0
        elas_3rd_456 = (6 * c_coeffs_3rd[7]) / 48.0

        elas3rd_file = open("ELADAT_3rd", mode="w")
        print("The third elastic constants.", file=elas3rd_file)
        print(
            "Please check the number of elastic constants for your structure.",
            file=elas3rd_file,
        )
        print("\n", file=elas3rd_file)

        print("C111 = ", format(elas_3rd_111, ".4f"), file=elas3rd_file)
        print("C112 = ", format(elas_3rd_112, ".4f"), file=elas3rd_file)
        print("C113 = ", format(elas_3rd_113, ".4f"), file=elas3rd_file)
        print("C123 = ", format(elas_3rd_123, ".4f"), file=elas3rd_file)
        print("\n", file=elas3rd_file)

        print("C144 = ", format(elas_3rd_144, ".4f"), file=elas3rd_file)
        print("C155 = ", format(elas_3rd_155, ".4f"), file=elas3rd_file)
        print("C166 = ", format(elas_3rd_166, ".4f"), file=elas3rd_file)
        print("C456 = ", format(elas_3rd_456, ".4f"), file=elas3rd_file)
        
        print("\n", end="", file=elas3rd_file)
        print("Please cite: Comput. Phys. Commun., 281 (2022), 108495", file=elas3rd_file)

    def __HexaI3rd_solve(self, c_coeffs_3rd=None, c_coeffs_2nd=None):

        elas_2nd = np.zeros((6, 6))

        elas_2nd[0, 0] = 2 * c_coeffs_2nd[0]
        elas_2nd[2, 2] = 2 * c_coeffs_2nd[1]
        elas_2nd[3, 3] = 2 * (c_coeffs_2nd[7] - c_coeffs_2nd[0]) / 4.0
        elas_2nd[0, 1] = 2 * (2 * c_coeffs_2nd[0] - c_coeffs_2nd[2]) / 2.0
        elas_2nd[0, 2] = 2 * (c_coeffs_2nd[1] + c_coeffs_2nd[0] - c_coeffs_2nd[4]) / 2.0

        elas_2nd[1, 1] = elas_2nd[0, 0]
        elas_2nd[1, 0] = elas_2nd[0, 1]
        elas_2nd[2, 0] = elas_2nd[1, 2] = elas_2nd[2, 1] = elas_2nd[0, 2]
        elas_2nd[4, 4] = elas_2nd[3, 3]
        elas_2nd[5, 5] = (elas_2nd[0, 0] - elas_2nd[0, 1]) / 2
        # print(elas_2nd)

        solve_elastic_3D.solve_elas3D().elasproperties(Celas=elas_2nd)

        elas_3rd_111 = 6 * c_coeffs_3rd[0]
        elas_3rd_222 = 6 * (4 * c_coeffs_3rd[0] - c_coeffs_3rd[2]) / 4.0
        elas_3rd_333 = 6 * c_coeffs_3rd[1]

        elas_3rd_112 = (
            6 * (c_coeffs_3rd[5] - 4 * c_coeffs_3rd[0] + 2 * c_coeffs_3rd[1]) / 6.0
        )
        elas_3rd_133 = (
            6 * (c_coeffs_3rd[3] + c_coeffs_3rd[4] - 2 * c_coeffs_3rd[0]) / 6.0
        )
        elas_3rd_113 = (
            -6 * (c_coeffs_3rd[4] - c_coeffs_3rd[0] + c_coeffs_3rd[1])
            + 3 * elas_3rd_133
        ) / 3.0
        elas_3rd_123 = (
            (-6.0) * (c_coeffs_3rd[6]) + elas_3rd_333 + 6 * elas_3rd_113
        ) / 6.0
        elas_3rd_144 = 6 * (c_coeffs_3rd[7] - c_coeffs_3rd[0]) / 12.0
        elas_3rd_155 = (6 * (c_coeffs_3rd[8]) - elas_3rd_222) / 12.0
        elas_3rd_344 = 6 * (c_coeffs_3rd[9] - c_coeffs_3rd[1]) / 12.0

        elas3rd_file = open("ELADAT_3rd", mode="w")
        print("The third elastic constants.", file=elas3rd_file)
        print(
            "Please check the number of elastic constants for your structure.",
            file=elas3rd_file,
        )
        print("\n", file=elas3rd_file)

        print("C111 = ", format(elas_3rd_111, ".4f"), file=elas3rd_file)
        print("C222 = ", format(elas_3rd_222, ".4f"), file=elas3rd_file)
        print("C333 = ", format(elas_3rd_333, ".4f"), file=elas3rd_file)

        print("\n", file=elas3rd_file)

        print("C112 = ", format(elas_3rd_112, ".4f"), file=elas3rd_file)
        print("C113 = ", format(elas_3rd_113, ".4f"), file=elas3rd_file)
        print("C133 = ", format(elas_3rd_133, ".4f"), file=elas3rd_file)
        print("C144 = ", format(elas_3rd_144, ".4f"), file=elas3rd_file)
        print("C155 = ", format(elas_3rd_155, ".4f"), file=elas3rd_file)
        print("C123 = ", format(elas_3rd_123, ".4f"), file=elas3rd_file)
        print("C344 = ", format(elas_3rd_344, ".4f"), file=elas3rd_file)
        
        print("\n", end="", file=elas3rd_file)
        print("Please cite: Comput. Phys. Commun., 281 (2022), 108495", file=elas3rd_file)

    def __HexaII3rd_solve(self, c_coeffs_3rd=None, c_coeffs_2nd=None):

        elas_2nd = np.zeros((6, 6))

        elas_2nd[0, 0] = 2 * c_coeffs_2nd[0]
        elas_2nd[2, 2] = 2 * c_coeffs_2nd[1]
        elas_2nd[3, 3] = 2 * (c_coeffs_2nd[7] - c_coeffs_2nd[0]) / 4.0
        elas_2nd[0, 1] = 2 * (2 * c_coeffs_2nd[0] - c_coeffs_2nd[2]) / 2.0
        elas_2nd[0, 2] = 2 * (c_coeffs_2nd[1] + c_coeffs_2nd[0] - c_coeffs_2nd[4]) / 2.0

        elas_2nd[1, 1] = elas_2nd[0, 0]
        elas_2nd[1, 0] = elas_2nd[0, 1]
        elas_2nd[2, 0] = elas_2nd[1, 2] = elas_2nd[2, 1] = elas_2nd[0, 2]
        elas_2nd[4, 4] = elas_2nd[3, 3]
        elas_2nd[5, 5] = (elas_2nd[0, 0] - elas_2nd[0, 1]) / 2
        # print(elas_2nd)

        solve_elastic_3D.solve_elas3D().elasproperties(Celas=elas_2nd)

        elas_3rd_111 = 6 * c_coeffs_3rd[0]
        elas_3rd_222 = 6 * (4 * c_coeffs_3rd[0] - c_coeffs_3rd[2]) / 4.0
        elas_3rd_333 = 6 * c_coeffs_3rd[1]

        elas_3rd_112 = (
            6 * (c_coeffs_3rd[5] - 4 * c_coeffs_3rd[0] + 2 * c_coeffs_3rd[1]) / 6.0
        )
        elas_3rd_133 = (
            6 * (c_coeffs_3rd[3] + c_coeffs_3rd[4] - 2 * c_coeffs_3rd[0]) / 6.0
        )
        elas_3rd_113 = (
            -6 * (c_coeffs_3rd[4] - c_coeffs_3rd[0] + c_coeffs_3rd[1])
            + 3 * elas_3rd_133
        ) / 3.0
        elas_3rd_123 = (
            (-6.0) * (c_coeffs_3rd[6]) + elas_3rd_333 + 6 * elas_3rd_113
        ) / 6.0
        elas_3rd_144 = 6 * (c_coeffs_3rd[7] - c_coeffs_3rd[0]) / 12.0
        elas_3rd_155 = (6 * (c_coeffs_3rd[8]) - elas_3rd_222) / 12.0
        elas_3rd_344 = 6 * (c_coeffs_3rd[9] - c_coeffs_3rd[1]) / 12.0

        elas_3rd_116 = 6 * (c_coeffs_3rd[10] - c_coeffs_3rd[0]) / 4.0
        elas_3rd_145 = (
            6 * (c_coeffs_3rd[11] - c_coeffs_3rd[0]) - elas_3rd_144 - elas_3rd_155
        ) / 8.0

        elas3rd_file = open("ELADAT_3rd", mode="w")
        print("The third elastic constants.", file=elas3rd_file)
        print(
            "Please check the number of elastic constants for your structure.",
            file=elas3rd_file,
        )
        print("\n", file=elas3rd_file)

        print("C111 = ", format(elas_3rd_111, ".4f"), file=elas3rd_file)
        print("C222 = ", format(elas_3rd_222, ".4f"), file=elas3rd_file)
        print("C333 = ", format(elas_3rd_333, ".4f"), file=elas3rd_file)

        print("\n", file=elas3rd_file)

        print("C112 = ", format(elas_3rd_112, ".4f"), file=elas3rd_file)
        print("C113 = ", format(elas_3rd_113, ".4f"), file=elas3rd_file)
        print("C116 = ", format(elas_3rd_116, ".4f"), file=elas3rd_file)
        print("C133 = ", format(elas_3rd_133, ".4f"), file=elas3rd_file)
        print("C144 = ", format(elas_3rd_144, ".4f"), file=elas3rd_file)
        print("C155 = ", format(elas_3rd_155, ".4f"), file=elas3rd_file)
        print("C123 = ", format(elas_3rd_123, ".4f"), file=elas3rd_file)
        print("C145 = ", format(elas_3rd_145, ".4f"), file=elas3rd_file)
        print("C344 = ", format(elas_3rd_344, ".4f"), file=elas3rd_file)
        
        print("\n", end="", file=elas3rd_file)
        print("Please cite: Comput. Phys. Commun., 281 (2022), 108495", file=elas3rd_file)

    def __orthorhombic3rd_solve(self, c_coeffs_3rd=None, c_coeffs_2nd=None):

        # The second elastics
        elas_2nd = np.zeros((6, 6))
        elas_2nd[0, 0] = c_coeffs_2nd[0]
        elas_2nd[1, 1] = c_coeffs_2nd[6]
        elas_2nd[2, 2] = c_coeffs_2nd[7]
        elas_2nd[3, 3] = 0.25 * (c_coeffs_2nd[3] - c_coeffs_2nd[0])
        elas_2nd[4, 4] = 0.25 * (c_coeffs_2nd[12] - c_coeffs_2nd[6])
        elas_2nd[5, 5] = 0.25 * (c_coeffs_2nd[4] - c_coeffs_2nd[0])
        elas_2nd[0, 1] = 0.5 * (c_coeffs_2nd[1] - c_coeffs_2nd[0] - c_coeffs_2nd[6])
        elas_2nd[0, 2] = 0.5 * (c_coeffs_2nd[19] - c_coeffs_2nd[0] - c_coeffs_2nd[7])
        elas_2nd[1, 2] = 0.5 * (c_coeffs_2nd[8] - c_coeffs_2nd[6] - c_coeffs_2nd[7])

        solve_elastic_3D.solve_elas3D().elasproperties(Celas=elas_2nd)

        # The third elastics
        elas_3rd_111 = c_coeffs_3rd[0]
        elas_3rd_222 = c_coeffs_3rd[6]
        elas_3rd_333 = c_coeffs_3rd[7]
        elas_3rd_456 = c_coeffs_3rd[5] / 48.0

        elas_3rd_144 = (c_coeffs_3rd[3] - c_coeffs_3rd[0]) / 12
        elas_3rd_155 = (c_coeffs_3rd[16] - c_coeffs_3rd[0]) / 12
        elas_3rd_166 = (c_coeffs_3rd[4] - c_coeffs_3rd[0]) / 12
        elas_3rd_244 = (c_coeffs_3rd[11] - c_coeffs_3rd[6]) / 12
        elas_3rd_255 = (c_coeffs_3rd[12] - c_coeffs_3rd[6]) / 12
        elas_3rd_266 = (c_coeffs_3rd[17] - c_coeffs_3rd[6]) / 12
        elas_3rd_344 = (c_coeffs_3rd[18] - c_coeffs_3rd[7]) / 12
        elas_3rd_355 = (c_coeffs_3rd[13] - c_coeffs_3rd[7]) / 12
        elas_3rd_366 = (c_coeffs_3rd[10] - c_coeffs_3rd[7]) / 12

        elas_3rd_112 = 0.5 * (
            c_coeffs_3rd[1]
            - 3 * c_coeffs_3rd[9]
            - (8 / 9) * c_coeffs_3rd[0]
            + 2 * c_coeffs_3rd[6]
        )
        elas_3rd_122 = (1 / 3) * (
            c_coeffs_3rd[1] - c_coeffs_3rd[0] - c_coeffs_3rd[6] - 3 * elas_3rd_112
        )

        elas_3rd_113 = 0.5 * (
            c_coeffs_3rd[19]
            - 3 * c_coeffs_3rd[15]
            - (8 / 9) * c_coeffs_3rd[0]
            + 2 * c_coeffs_3rd[7]
        )
        elas_3rd_133 = (1 / 3) * (
            c_coeffs_3rd[19] - c_coeffs_3rd[0] - c_coeffs_3rd[7] - 3 * elas_3rd_113
        )

        elas_3rd_223 = 0.5 * (
            c_coeffs_3rd[8]
            - 3 * c_coeffs_3rd[14]
            - (8 / 9) * c_coeffs_3rd[6]
            + 2 * c_coeffs_3rd[7]
        )
        elas_3rd_233 = (1 / 3) * (
            c_coeffs_3rd[8] - c_coeffs_3rd[6] - c_coeffs_3rd[7] - 3 * elas_3rd_223
        )

        elas_3rd_123 = (1 / 6) * (
            c_coeffs_3rd[2]
            - c_coeffs_3rd[0]
            - c_coeffs_3rd[6]
            - c_coeffs_3rd[7]
            - 3
            * (
                elas_3rd_112
                + elas_3rd_122
                + elas_3rd_113
                + elas_3rd_133
                + elas_3rd_223
                + elas_3rd_233
            )
        )

        elas3rd_file = open("ELADAT_3rd", mode="w")
        print("The third elastic constants.", file=elas3rd_file)
        print(
            "Please check the number of elastic constants for your structure.",
            file=elas3rd_file,
        )
        print("\n", file=elas3rd_file)

        print("C111 = ", format(elas_3rd_111, ".4f"), file=elas3rd_file)
        print("C222 = ", format(elas_3rd_222, ".4f"), file=elas3rd_file)
        print("C333 = ", format(elas_3rd_333, ".4f"), file=elas3rd_file)
        print("\n", file=elas3rd_file)

        print("C112 = ", format(elas_3rd_112, ".4f"), file=elas3rd_file)
        print("C113 = ", format(elas_3rd_113, ".4f"), file=elas3rd_file)
        print("C122 = ", format(elas_3rd_122, ".4f"), file=elas3rd_file)
        print("C133 = ", format(elas_3rd_133, ".4f"), file=elas3rd_file)
        print("C144 = ", format(elas_3rd_144, ".4f"), file=elas3rd_file)
        print("C155 = ", format(elas_3rd_155, ".4f"), file=elas3rd_file)
        print("C166 = ", format(elas_3rd_166, ".4f"), file=elas3rd_file)
        print("\n", file=elas3rd_file)

        print("C223 = ", format(elas_3rd_223, ".4f"), file=elas3rd_file)
        print("C233 = ", format(elas_3rd_233, ".4f"), file=elas3rd_file)
        print("C244 = ", format(elas_3rd_244, ".4f"), file=elas3rd_file)
        print("C255 = ", format(elas_3rd_255, ".4f"), file=elas3rd_file)
        print("C266 = ", format(elas_3rd_266, ".4f"), file=elas3rd_file)
        print("\n", file=elas3rd_file)

        print("C344 = ", format(elas_3rd_344, ".4f"), file=elas3rd_file)
        print("C355 = ", format(elas_3rd_355, ".4f"), file=elas3rd_file)
        print("C366 = ", format(elas_3rd_366, ".4f"), file=elas3rd_file)
        print("\n", file=elas3rd_file)

        print("C123 = ", format(elas_3rd_123, ".4f"), file=elas3rd_file)
        print("C456 = ", format(elas_3rd_456, ".4f"), file=elas3rd_file)
        print("\n", end="", file=elas3rd_file)
        print("Please cite: Comput. Phys. Commun., 281 (2022), 108495", file=elas3rd_file)

    def __tetra3rd_solve(self, c_coeffs_3rd=None, c_coeffs_2nd=None):

        # The second elastics
        elas_2nd = np.zeros((6, 6))
        elas_2nd[0, 0] = c_coeffs_2nd[0]
        elas_2nd[2, 2] = c_coeffs_2nd[7]
        elas_2nd[3, 3] = 0.25 * (c_coeffs_2nd[3] - c_coeffs_2nd[0])
        elas_2nd[4, 4] = 0.25 * (c_coeffs_2nd[16] - c_coeffs_2nd[0])
        elas_2nd[5, 5] = 0.25 * (c_coeffs_2nd[4] - c_coeffs_2nd[0])
        elas_2nd[0, 1] = 0.5 * (c_coeffs_2nd[1] - 2 * c_coeffs_2nd[0])
        elas_2nd[0, 2] = 0.25 * (
            c_coeffs_2nd[2] - 2 * c_coeffs_2nd[0] - c_coeffs_2nd[7] - 2 * elas_2nd[0, 1]
        )
        elas_2nd[1, 1] = c_coeffs_2nd[8] - c_coeffs_2nd[7] - 2 * elas_2nd[0, 2]

        solve_elastic_3D.solve_elas3D().elasproperties(Celas=elas_2nd)

        # The third elastics
        elas_3rd_111 = c_coeffs_3rd[0]
        elas_3rd_333 = c_coeffs_3rd[7]

        elas_3rd_112 = (1 / 6) * (c_coeffs_3rd[1] - 2 * c_coeffs_3rd[0])
        elas_3rd_113 = 0.5 * (
            c_coeffs_3rd[8]
            - 3 * c_coeffs_3rd[15]
            - (8 / 9) * c_coeffs_3rd[0]
            + 2 * c_coeffs_3rd[7]
        )
        elas_3rd_133 = (1 / 3) * (
            c_coeffs_3rd[8] - c_coeffs_3rd[0] - c_coeffs_3rd[7] - 3 * elas_3rd_113
        )
        elas_3rd_144 = (1 / 12) * (c_coeffs_3rd[3] - c_coeffs_3rd[0])
        elas_3rd_155 = (1 / 12) * (c_coeffs_3rd[16] - c_coeffs_3rd[0])
        elas_3rd_166 = (1 / 12) * (c_coeffs_3rd[4] - c_coeffs_3rd[0])

        elas_3rd_344 = (1 / 12) * (c_coeffs_3rd[18] - c_coeffs_3rd[7])
        elas_3rd_366 = (1 / 12) * (c_coeffs_3rd[10] - c_coeffs_3rd[7])

        elas_3rd_123 = (1 / 6) * (
            c_coeffs_3rd[2]
            - 2 * c_coeffs_3rd[0]
            - c_coeffs_3rd[7]
            - 6 * (elas_3rd_112 + elas_3rd_113 + elas_3rd_133)
        )
        elas_3rd_456 = (1 / 48) * c_coeffs_3rd[5]

        elas3rd_file = open("ELADAT_3rd", mode="w")
        print("The third elastic constants.", file=elas3rd_file)
        print(
            "Please check the number of elastic constants for your structure.",
            file=elas3rd_file,
        )
        print("\n", file=elas3rd_file)

        print("C111 = ", format(elas_3rd_111, ".4f"), file=elas3rd_file)
        # print("C222 = ", format(elas_3rd_222, ".4f"), file=elas3rd_file)
        print("C333 = ", format(elas_3rd_333, ".4f"), file=elas3rd_file)
        print("\n", file=elas3rd_file)

        print("C112 = ", format(elas_3rd_112, ".4f"), file=elas3rd_file)
        print("C113 = ", format(elas_3rd_113, ".4f"), file=elas3rd_file)
        # print("C122 = ", format(elas_3rd_122, ".4f"), file=elas3rd_file)
        print("C133 = ", format(elas_3rd_133, ".4f"), file=elas3rd_file)
        print("C144 = ", format(elas_3rd_144, ".4f"), file=elas3rd_file)
        print("C155 = ", format(elas_3rd_155, ".4f"), file=elas3rd_file)
        print("C166 = ", format(elas_3rd_166, ".4f"), file=elas3rd_file)
        print("\n", file=elas3rd_file)

        # print("C223 = ", format(elas_3rd_223, ".4f"), file=elas3rd_file)
        # print("C233 = ", format(elas_3rd_233, ".4f"), file=elas3rd_file)
        # print("C244 = ", format(elas_3rd_244, ".4f"), file=elas3rd_file)
        # print("C255 = ", format(elas_3rd_255, ".4f"), file=elas3rd_file)
        # print("C266 = ", format(elas_3rd_266, ".4f"), file=elas3rd_file)
        # print('\n', file=elas3rd_file)

        print("C344 = ", format(elas_3rd_344, ".4f"), file=elas3rd_file)
        # print("C355 = ", format(elas_3rd_355, ".4f"), file=elas3rd_file)
        print("C366 = ", format(elas_3rd_366, ".4f"), file=elas3rd_file)
        print("\n", file=elas3rd_file)

        print("C123 = ", format(elas_3rd_123, ".4f"), file=elas3rd_file)
        print("C456 = ", format(elas_3rd_456, ".4f"), file=elas3rd_file)
        
        print("\n", end="", file=elas3rd_file)
        print("Please cite: Comput. Phys. Commun., 281 (2022), 108495", file=elas3rd_file)

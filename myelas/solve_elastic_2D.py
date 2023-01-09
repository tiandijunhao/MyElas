import os
import numpy as np

from . import read_poscar as readpos
from . import standard_cell
from . import plot_fit_E
from . import strain_matrix_string


class solve_elas2D(object):
    def __init__(self):
        print("Solve 2D elastic constants.")
        self.spg_num = readpos.read_poscar().spacegroup_num()
        self.lattindex = readpos.read_poscar().latt_index()
        self.latt = readpos.read_poscar().latti()
        atom_num = standard_cell.recell(to_pricell=False).atom_number()

        recipvect = np.array([[0.0, 0.0, 0.0]])
        recipvect[0, 0] = (
            self.latt[0, 1] * self.latt[1, 2] - self.latt[1, 1] * self.latt[0, 2]
        )
        recipvect[0, 1] = (
            self.latt[0, 2] * self.latt[1, 0] - self.latt[1, 2] * self.latt[0, 0]
        )
        recipvect[0, 2] = (
            self.latt[0, 0] * self.latt[1, 1] - self.latt[1, 0] * self.latt[0, 1]
        )

        self.area = (self.lattindex ** 2.0) * np.sqrt(
            recipvect[0, 0] ** 2 + recipvect[0, 1] ** 2 + recipvect[0, 2] ** 2
        )

    def solve(
        self, strain_max=None, strain_num=None,
    ):
        """
        To solve the second elastics
        strain_max : must be equal to the maxinum of generating strain poscar.
        strain_num : must be equal to the number of strain poscar in one elastic independent.
        """

        if self.spg_num >= 3 and self.spg_num <= 15:
            nelastic = 6
        else:
            nelastic = 4
        
        strain_matrix_str=strain_matrix_string.Elastics_3D(nelastic=nelastic)

        starin_step = 2.0 * strain_max / (strain_num - 1)
        strain_param = np.arange(-strain_max, strain_max + 0.0001, starin_step)

        strain_energy = {}
        fit_coeffs = []

        E_strain_data = open("E_Strain.out", mode="w")
        for nelas in np.arange(0, nelastic, 1):
            print(
                "#2D_nelastic {:0>2d}  strain  energy  (E-E0)/V0".format(nelas),
                file=E_strain_data,
            )
            energy = []
            fit_energy = []
            for ndef in np.arange(0, strain_num, 1):
                full_path = (
                    "2D_nelastic_"
                    + str(format(nelas + 1, "02d"))
                    + "/strain_"
                    + str(format(ndef + 1, "03d"))
                )

                # readout = open("OSZICAR_"+str(format(nelas + 1, "02d")) +
                #        "_"+str(format(ndef + 1, "03d")), mode="r",)
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

                fit_param = (np.array(energy)[j] - np.array(energy).min()) / (self.area)
                fit_energy.append(fit_param)

                print(
                    "{:.6f}  {:.8f}  {:.8f}".format(
                        strain_param[j], energy[j], fit_param
                    ),
                    file=E_strain_data,
                )

            fit_poly = np.polyfit(strain_param, np.array(fit_energy), 3)
            fit_func = np.poly1d(fit_poly)
            coffs = fit_func.coeffs * 16.021766208
            fit_coeffs.append(coffs[1])

            # plot fit function
            plot_fit_E.plot_energy_fit().plot_energy(
                x=strain_param,
                y=fit_energy,
                fit_func=fit_func,
                figname="Nelastic {:0>1d}".format(nelas+1),
                strain_str = strain_matrix_str[nelas]
            )

        self.__rect2d_solve(c_coeffs=fit_coeffs)

    def __rect2d_solve(self, c_coeffs=None):
        """
        2D elastic tensor
        ----------------------------
        C11  C12  C16 \\
        C12  C22  C26 \\
        C16  C26  C66
        """

        elas = np.zeros((3, 3))

        elas[0, 0] = 2 * c_coeffs[0]
        elas[1, 1] = 2 * c_coeffs[1]
        elas[2, 2] = 2 * c_coeffs[2]

        elas[0, 1] = c_coeffs[3] - c_coeffs[0] - c_coeffs[1]
        elas[1, 0] = elas[0, 1]
        if self.spg_num >= 3 and self.spg_num <= 15:
            elas[0, 2] = c_coeffs[4] - c_coeffs[0] - c_coeffs[4]
            elas[2, 0] = elas[0, 2]

            elas[1, 2] = c_coeffs[5] - c_coeffs[1] - c_coeffs[4]
            elas[2, 1] = elas[1, 2]

        Title = "The 2D mechanical properties"

        self.elasproperties(Celas=elas, title=Title)

    def elasproperties(self, Celas=None, title=None):

        Selas = np.linalg.inv(Celas)

        B_v = (Celas[0, 0] + Celas[1, 1] + 2 * Celas[0, 1]) / 4.0
        G_v = (Celas[0, 0] + Celas[1, 1] - 2 * Celas[0, 1] + 4 * Celas[2, 2]) / 8.0

        B_r = 1 / (Selas[0, 0] + Selas[1, 1] + 2 * Selas[0, 1])
        G_r = 2 / (Selas[0, 0] + Selas[1, 1] - 2 * Selas[0, 1] + Selas[2, 2])

        # 2D Young's modulus
        Ex = (Celas[0, 0] * Celas[1, 1] - Celas[1, 0] * Celas[0, 1]) / Celas[1, 1]
        Ey = (Celas[0, 0] * Celas[1, 1] - Celas[1, 0] * Celas[0, 1]) / Celas[0, 0]
        Gxy = Celas[2, 2]

        # 2D Poisson ratio
        Muxy = Celas[1, 0] / Celas[1, 1]
        Muyx = Celas[0, 1] / Celas[0, 0]

        # Anisotropy indices
        ## the elastic anisotropy
        A_SU = np.sqrt((B_v / B_r - 1) ** 2 + 2 * (G_v / G_r - 1) ** 2)

        ## Ranganathan
        A_R = B_v / B_r + 2 * G_v / G_r - 3

        ## Kube
        A_K = np.sqrt((np.log(B_v / B_r)) ** 2 + 2 * (np.log(G_v / G_r)) ** 2)

        if Celas[2, 2] > 0 and (Celas[0, 0] * Celas[1, 1] - Celas[0, 1] ** 2) > 0:
            if_stable = "Stable"
        else:
            if_stable = "Unstable"

        elasfile = open("second_elastic.out", mode="w")

        print(
            "2D elastic constants", file=elasfile,
        )
        print("Space group: ", format(self.spg_num), file=elasfile)

        print("\n", end="", file=elasfile)
        print("Elastic tensor C_ij (unit: GPa)", file=elasfile)

        for i in np.arange(0, 3, 1):
            print(
                "  ",
                format(Celas[i, 0], ".3f"),
                "  ",
                format(Celas[i, 1], ".3f"),
                "  ",
                format(Celas[i, 2], ".3f"),
                file=elasfile,
            )
        print("\n", end="", file=elasfile)
        print("Compliance tensor S_ij (unit: GPa^-1)", file=elasfile)

        for i in np.arange(0, 3, 1):
            print(
                "  ",
                format(Selas[i, 0], ".6f"),
                "  ",
                format(Selas[i, 1], ".6f"),
                "  ",
                format(Selas[i, 2], ".6f"),
                file=elasfile,
            )

        print("\n", end="", file=elasfile)

        print("2D area (m^2) :", format(self.area, ".4f"), file=elasfile)
        print("\n", end="", file=elasfile)

        print(
            "Young(Ex and Ey) and shear(Gxy) moduli", file=elasfile,
        )
        print("Ex : {:.4f}".format(Ex), file=elasfile)
        print("Ey : {:.4f}".format(Ey), file=elasfile)
        print("Gxy: {:.4f}".format(Gxy), file=elasfile)

        print("\n", end="", file=elasfile)
        print(
            "Poisson ratios(Muxy and Muyx)", file=elasfile,
        )
        print("Muxy : {:.4f}".format(Muxy), file=elasfile)
        print("Muyx : {:.4f}".format(Muyx), file=elasfile)

        print("\n", end="", file=elasfile)
        print("mechanical stability:  " + if_stable, file=elasfile)

        print("\n", end="", file=elasfile)

        print("Anisotropy index:", file=elasfile)
        print("  Elastic anisotropy index      : ", format(A_SU, ".2f"), file=elasfile)
        print("  Ranganathan anisotropy index  : ", format(A_R, ".2f"), file=elasfile)
        print("  Kube anisotropy index         : ", format(A_K, ".2f"), file=elasfile)
        
        print("\n", end="", file=elasfile)
        print("Please cite: Comput. Phys. Commun., 281 (2022), 108495", file=elasfile)

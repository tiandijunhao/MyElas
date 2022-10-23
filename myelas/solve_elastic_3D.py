import os
import numpy as np

from . import read_poscar as readpos
from . import standard_cell
from . import plot_fit_E
from . import strain_matrix_string


# import read_poscar as readpos
# import standard_cell


class solve_elas3D(object):
    def __init__(
        self,
        spg_num=None,
        lattindex=None,
        latt=None,
        volume=None,
        density=None,
        mass=None,
        num=None,
    ):

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
        self, strain_max=None, strain_num=None, theta_degree=None, phi_degree=None,
    ):
        """
        To solve the second elastics
        -------
        strain_max : must be equal to the maxinum of generating strain poscar.
        strain_num : must be equal to the number of strain poscar in one elastic independent.
        """

        if self.spg_num >= 1 and self.spg_num <= 2:
            nelastic = 21

        elif self.spg_num >= 3 and self.spg_num <= 15:
            nelastic = 13

        elif self.spg_num >= 16 and self.spg_num <= 74:
            nelastic = 9

        elif self.spg_num >= 75 and self.spg_num <= 88:
            nelastic = 7

        elif self.spg_num >= 89 and self.spg_num <= 142:
            nelastic = 6

        elif self.spg_num >= 143 and self.spg_num <= 148:
            nelastic = 8

        elif self.spg_num >= 149 and self.spg_num <= 167:
            nelastic = 6

        elif self.spg_num >= 168 and self.spg_num <= 194:
            nelastic = 5

        elif self.spg_num >= 195 and self.spg_num <= 230:
            nelastic = 3

        strain_matrix_str=strain_matrix_string.Elastics_3D(nelastic=nelastic)
        
        starin_step = 2.0 * strain_max / (strain_num - 1)
        strain_param = np.arange(-strain_max, strain_max + 0.0001, starin_step)

        strain_energy = {}
        fit_coeffs = []

        # Read and save the strain energy
        E_strain_data = open("E_Strain.out", mode="w")
        print("Space group {:0>3d}".format(self.spg_num), file=E_strain_data)

        for nelas in np.arange(0, nelastic, 1):
            print(
                "#nelastic {:0>2d}  strain  energy  (E-E0)/V0".format(nelas),
                file=E_strain_data,
            )

            energy = []
            fit_energy = []
            for ndef in np.arange(0, strain_num, 1):
                full_path = (
                    "nelastic_"
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

            fit_poly = np.polyfit(strain_param, np.array(fit_energy), 3)
            fit_func = np.poly1d(fit_poly)
            coffs = fit_func.coeffs * 160.21766208
            fit_coeffs.append(coffs[1])

            # plot fit function
            plot_fit_E.plot_energy_fit().plot_energy(
                x=strain_param,
                y=fit_energy,
                fit_func=fit_func,
                figname="Nelastic {:0>1d}".format(nelas + 1),
                strain_str = strain_matrix_str[nelas]
            )

        if self.spg_num >= 1 and self.spg_num <= 2:
            self.__triclinic_solve(
                c_coeffs=fit_coeffs, theta=theta_degree, phi=phi_degree
            )

        elif self.spg_num >= 3 and self.spg_num <= 15:
            self.__monoclinic_solve(
                c_coeffs=fit_coeffs, theta=theta_degree, phi=phi_degree
            )

        elif self.spg_num >= 16 and self.spg_num <= 74:
            self.__orthorhombic_solve(
                c_coeffs=fit_coeffs, theta=theta_degree, phi=phi_degree
            )

        elif self.spg_num >= 75 and self.spg_num <= 88:
            self.__tetragonal_II_solve(
                c_coeffs=fit_coeffs, theta=theta_degree, phi=phi_degree
            )

        elif self.spg_num >= 89 and self.spg_num <= 142:
            self.__tetragonal_I_solve(
                c_coeffs=fit_coeffs, theta=theta_degree, phi=phi_degree
            )

        elif self.spg_num >= 143 and self.spg_num <= 148:
            self.__rhombohedral_II_solve(
                c_coeffs=fit_coeffs, theta=theta_degree, phi=phi_degree
            )

        elif self.spg_num >= 149 and self.spg_num <= 167:
            self.__rhombohedral_I_solve(
                c_coeffs=fit_coeffs, theta=theta_degree, phi=phi_degree
            )

        elif self.spg_num >= 168 and self.spg_num <= 194:
            self.__hexagonal_solve(
                c_coeffs=fit_coeffs, theta=theta_degree, phi=phi_degree
            )

        elif self.spg_num >= 195 and self.spg_num <= 230:
            self.__cubic_solve(c_coeffs=fit_coeffs, theta=theta_degree, phi=phi_degree)

    def __cubic_solve(self, c_coeffs=None, theta=None, phi=None):
        """
        Cubic elastic tensor
        ----------------------------
        C11  C12  C12  0    0    0 \\
        C12  C11  C12  0    0    0 \\
        C12  C12  C11  0    0    0 \\
        0    0    0    C44  0    0 \\
        0    0    0    0    C44  0 \\
        0    0    0    0    0    C44
        """

        elas = np.zeros((6, 6))

        elas[3, 3] = (2.0 / 3.0) * c_coeffs[0]
        elas[0, 1] = (2.0 / 3.0) * c_coeffs[2] - c_coeffs[1]
        elas[0, 0] = c_coeffs[1] - elas[0, 1]

        elas[4, 4] = elas[5, 5] = elas[3, 3]
        elas[1, 0] = elas[0, 2] = elas[2, 0] = elas[1, 2] = elas[2, 1] = elas[0, 1]
        elas[1, 1] = elas[2, 2] = elas[0, 0]

        Title = "The cubic crystal mechanical properties"

        self.elasproperties(Celas=elas, title=Title, theta=theta, phi=phi)

    def __hexagonal_solve(self, c_coeffs=None, theta=None, phi=None):
        """
        Hexagonal elastic tensor
        ----------------------------
        C11  C12  C13  0    0    0 \\
        C12  C11  C13  0    0    0 \\
        C13  C13  C33  0    0    0 \\
        0    0    0    C44  0    0 \\
        0    0    0    0    C44  0 \\
        0    0    0    0    0    (C11-C12)/2
        """

        elas = np.zeros((6, 6))

        elas[0, 0] = (4.0 * c_coeffs[1] + c_coeffs[0]) / 2.0
        elas[0, 1] = c_coeffs[0] - elas[0, 0]
        elas[2, 2] = 2.0 * c_coeffs[2]
        elas[3, 3] = c_coeffs[3]
        elas[0, 2] = (c_coeffs[4] - elas[0, 0] - elas[0, 1] - elas[2, 2] / 2.0) / 2.0

        elas[1, 1] = elas[0, 0]
        elas[1, 0] = elas[0, 1]
        elas[2, 0] = elas[1, 2] = elas[2, 1] = elas[0, 2]
        elas[4, 4] = elas[3, 3]
        elas[5, 5] = (elas[0, 0] - elas[0, 1]) / 2.0

        Title = "The hexagonal crystal mechanical properties"

        self.elasproperties(Celas=elas, title=Title, theta=theta, phi=phi)

    def __rhombohedral_I_solve(self, c_coeffs=None, theta=None, phi=None):
        """
        Rhombohedral I elastic tensor
        ----------------------------
        C11  C12  C13  C14  0    0 \\
        C12  C11  C13 -C14  0    0 \\
        C13  C13  C33  0    0    0 \\
        C14 -C14  0    C44  0    0 \\
        0    0    0    0    C44  C14 \\
        0    0    0    0    C14  (C11-C12)/2
        """

        elas = np.zeros((6, 6))

        elas[0, 0] = (4.0 * c_coeffs[1] + c_coeffs[0]) / 2.0
        elas[0, 1] = c_coeffs[0] - elas[0, 0]
        elas[2, 2] = 2.0 * c_coeffs[2]
        elas[3, 3] = c_coeffs[3]
        elas[0, 2] = (c_coeffs[4] - elas[0, 0] - elas[0, 1] - elas[2, 2] / 2.0) / 2.0
        elas[0, 3] = c_coeffs[5] - c_coeffs[3] / 2.0 - c_coeffs[1]

        elas[1, 1] = elas[0, 0]
        elas[1, 0] = elas[0, 1]
        elas[2, 0] = elas[1, 2] = elas[2, 1] = elas[0, 2]
        elas[4, 4] = elas[3, 3]
        elas[5, 5] = (elas[0, 0] - elas[0, 1]) / 2.0
        elas[3, 0] = elas[4, 5] = elas[5, 4] = elas[0, 3]
        elas[1, 3] = elas[3, 1] = -elas[0, 3]

        Title = "The rhombohedral I crystal mechanical properties"

        self.elasproperties(Celas=elas, title=Title, theta=theta, phi=phi)

    def __rhombohedral_II_solve(self, c_coeffs=None, theta=None, phi=None):
        """
        Rhombohedral II elastic tensor
        ----------------------------
        C11  C12  C13  C14  C15  0 \\
        C12  C11  C13 -C14 -C15  0 \\
        C13  C13  C33  0    0    0 \\
        C14 -C14  0    C44  0   -C45 \\
        C15 -C15  0    0    C44  C14 \\
        0    0    0   -C45  C14  (C11-C12)/2
        """

        elas = np.zeros((6, 6))

        elas[0, 0] = (4.0 * c_coeffs[1] + c_coeffs[0]) / 2.0
        elas[0, 1] = c_coeffs[0] - elas[0, 0]
        elas[2, 2] = 2.0 * c_coeffs[2]
        elas[3, 3] = c_coeffs[3]
        elas[0, 2] = (c_coeffs[4] - elas[0, 0] - elas[0, 1] - elas[2, 2] / 2.0) / 2.0
        elas[0, 3] = c_coeffs[5]
        elas[0, 4] = -c_coeffs[6]
        elas[3, 5] = c_coeffs[7]

        elas[1, 1] = elas[0, 0]
        elas[1, 0] = elas[0, 1]
        elas[2, 0] = elas[1, 2] = elas[2, 1] = elas[0, 2]
        elas[4, 4] = elas[3, 3]
        elas[5, 5] = (elas[0, 0] - elas[0, 1]) / 2.0
        elas[3, 0] = elas[4, 5] = elas[5, 4] = elas[0, 3]
        elas[1, 3] = elas[3, 1] = -elas[0, 3]
        elas[1, 4] = elas[4, 1] = c_coeffs[7]
        elas[4, 0] = elas[0, 4]
        elas[5, 3] = elas[3, 5]

        Title = "The rhombohedral II crystal mechanical properties"

        self.elasproperties(Celas=elas, title=Title, theta=theta, phi=phi)

    def __tetragonal_I_solve(self, c_coeffs=None, theta=None, phi=None):
        """
        Tetragonal I elastic tensor
        ----------------------------
        C11  C12  C13  0    0    0 \\
        C12  C11  C13  0    0    0 \\
        C13  C13  C33  0    0    0 \\
        0    0    0    C44  0    0 \\
        0    0    0    0    C44  0 \\
        0    0    0    0    0    C66 
        """

        elas = np.zeros((6, 6))

        elas[5, 5] = 2.0 * c_coeffs[1]
        elas[2, 2] = 2.0 * c_coeffs[2]
        elas[3, 3] = c_coeffs[3]
        elas[0, 1] = c_coeffs[4] - 2.0 * c_coeffs[5] + c_coeffs[2]
        elas[0, 0] = c_coeffs[0] - elas[0, 1]
        elas[0, 2] = c_coeffs[5] - elas[0, 0] / 2.0 - c_coeffs[2]

        elas[1, 1] = elas[0, 0]
        elas[1, 0] = elas[0, 1]
        elas[2, 0] = elas[1, 2] = elas[2, 1] = elas[0, 2]
        elas[4, 4] = elas[3, 3]

        Title = "The tetragonal I crystal mechanical properties"

        self.elasproperties(Celas=elas, title=Title, theta=theta, phi=phi)

    def __tetragonal_II_solve(self, c_coeffs=None, theta=None, phi=None):
        """
        Tetragonal II elastic tensor
        ----------------------------
        C11  C12  C13  0    0    C16 \\
        C12  C11  C13  0    0   -C16 \\
        C13  C13  C33  0    0    0 \\
        0    0    0    C44  0    0 \\
        0    0    0    0    C44  0 \\
        C16 -C16  0    0    0    C66 \\
        """

        elas = np.zeros((6, 6))

        elas[0, 0] = c_coeffs[0] - (c_coeffs[4] - 2.0 * c_coeffs[5] + c_coeffs[2])
        elas[1, 1] = elas[0, 0]
        elas[2, 2] = 2.0 * c_coeffs[2]
        elas[3, 3] = c_coeffs[3]
        elas[4, 4] = elas[3, 3]
        elas[5, 5] = 2.0 * c_coeffs[1]

        elas[0, 1] = c_coeffs[4] - 2.0 * c_coeffs[5] + c_coeffs[2]
        elas[0, 2] = (c_coeffs[4] - c_coeffs[2] - c_coeffs[0]) / 2.0
        elas[0, 5] = c_coeffs[6] - elas[0, 0] / 2.0 - elas[5, 5] / 2.0
        elas[1, 2] = elas[0, 2]
        elas[1, 5] = -elas[0, 5]

        elas[1, 0] = elas[0, 1]
        elas[2, 1] = elas[1, 2]
        elas[2, 0] = elas[0, 2]
        elas[5, 1] = elas[0, 5]
        elas[5, 1] = elas[1, 5]

        Title = "The tetragonal II crystal mechanical properties"

        self.elasproperties(Celas=elas, title=Title, theta=theta, phi=phi)

    def __orthorhombic_solve(self, c_coeffs=None, theta=None, phi=None):
        """
        Orthorhombic elastic tensor
        ----------------------------
        C11  C12  C13  0    0    0 \\
        C12  C22  C23  0    0    0 \\
        C13  C23  C33  0    0    0 \\
        0    0    0    C44  0    0 \\
        0    0    0    0    C55  0 \\
        0    0    0    0    0    C66
        """

        elas = np.zeros((6, 6))

        elas[0, 0] = 2 * c_coeffs[0]
        elas[1, 1] = 2 * c_coeffs[1]
        elas[2, 2] = 2 * c_coeffs[2]
        elas[3, 3] = 2 * c_coeffs[3]
        elas[4, 4] = 2 * c_coeffs[4]
        elas[5, 5] = 2 * c_coeffs[5]

        elas[0, 1] = c_coeffs[6] - c_coeffs[0] - c_coeffs[1]
        elas[1, 2] = c_coeffs[7] - c_coeffs[1] - c_coeffs[2]
        elas[0, 2] = c_coeffs[8] - c_coeffs[0] - c_coeffs[2]

        elas[1, 0] = elas[0, 1]
        elas[2, 1] = elas[1, 2]
        elas[2, 0] = elas[0, 2]

        Title = "The orthorhombic crystal mechanical properties"

        self.elasproperties(Celas=elas, title=Title, theta=theta, phi=phi)

    def __monoclinic_solve(self, c_coeffs, theta=None, phi=None):

        """
        Monoclinic elastic tensor
        ----------------------------
        C11  C12  C13  0    C15    0 \\
        C12  C22  C23  0    C25    0 \\
        C13  C23  C33  0    C35    0 \\
        0    0    0    C44  0    C46 \\
        C15  C25  C35  0    C55  0 \\
        0    0    0    C46  0    C66
        """

        elas = np.zeros((6, 6))

        elas[0, 0] = 2 * c_coeffs[0]
        elas[1, 1] = 2 * c_coeffs[1]
        elas[2, 2] = 2 * c_coeffs[2]
        elas[3, 3] = 2 * c_coeffs[3]
        elas[4, 4] = 2 * c_coeffs[4]
        elas[5, 5] = 2 * c_coeffs[5]

        elas[0, 1] = c_coeffs[6] - c_coeffs[0] - c_coeffs[1]
        elas[1, 2] = c_coeffs[7] - c_coeffs[1] - c_coeffs[2]
        elas[0, 2] = c_coeffs[8] - c_coeffs[0] - c_coeffs[2]
        elas[0, 4] = c_coeffs[9] - c_coeffs[0] - c_coeffs[4]
        elas[1, 4] = c_coeffs[10] - c_coeffs[1] - c_coeffs[4]
        elas[2, 4] = c_coeffs[11] - c_coeffs[2] - c_coeffs[4]
        elas[3, 5] = c_coeffs[12] - c_coeffs[3] - c_coeffs[5]

        elas[1, 0] = elas[0, 1]
        elas[2, 1] = elas[1, 2]
        elas[2, 0] = elas[0, 2]
        elas[4, 0] = elas[0, 4]
        elas[4, 1] = elas[1, 4]
        elas[4, 2] = elas[2, 4]
        elas[5, 3] = elas[3, 5]

        Title = "The monoclinic crystal mechanical properties"

        self.elasproperties(Celas=elas, title=Title, theta=theta, phi=phi)

    def __triclinic_solve(self, c_coeffs, theta=None, phi=None):
        """
        Triclinic elastic tensor
        ----------------------------
        C11  C12  C13  C14  C15  C16 \\
        C12  C22  C23  C24  C25  C26 \\
        C13  C23  C33  C34  C35  C36 \\
        C14  C24  C34  C44  C45  C46 \\
        C15  C25  C35  C45  C55  C56 \\
        C16  C26  C36  C46  C45  C66
        """

        elas = np.zeros((6, 6))

        elas[0, 0] = 2 * c_coeffs[0]
        elas[1, 1] = 2 * c_coeffs[1]
        elas[2, 2] = 2 * c_coeffs[2]
        elas[3, 3] = 2 * c_coeffs[3]
        elas[4, 4] = 2 * c_coeffs[4]
        elas[5, 5] = 2 * c_coeffs[5]

        elas[0, 1] = c_coeffs[6] - c_coeffs[0] - c_coeffs[1]
        elas[0, 2] = c_coeffs[7] - c_coeffs[0] - c_coeffs[2]
        elas[0, 3] = c_coeffs[8] - c_coeffs[0] - c_coeffs[3]
        elas[0, 4] = c_coeffs[9] - c_coeffs[0] - c_coeffs[4]
        elas[0, 5] = c_coeffs[10] - c_coeffs[0] - c_coeffs[5]
        elas[1, 2] = c_coeffs[11] - c_coeffs[1] - c_coeffs[2]
        elas[1, 3] = c_coeffs[12] - c_coeffs[1] - c_coeffs[3]
        elas[1, 4] = c_coeffs[13] - c_coeffs[1] - c_coeffs[4]
        elas[1, 5] = c_coeffs[14] - c_coeffs[1] - c_coeffs[5]
        elas[2, 3] = c_coeffs[15] - c_coeffs[2] - c_coeffs[3]
        elas[2, 4] = c_coeffs[16] - c_coeffs[2] - c_coeffs[4]
        elas[2, 5] = c_coeffs[17] - c_coeffs[2] - c_coeffs[5]
        elas[3, 4] = c_coeffs[18] - c_coeffs[3] - c_coeffs[4]
        elas[3, 5] = c_coeffs[19] - c_coeffs[3] - c_coeffs[5]
        elas[4, 5] = c_coeffs[20] - c_coeffs[4] - c_coeffs[5]

        elas[1, 0] = elas[0, 1]
        elas[2, 0] = elas[0, 2]
        elas[3, 0] = elas[0, 3]
        elas[4, 0] = elas[0, 4]
        elas[5, 0] = elas[0, 5]
        elas[2, 1] = elas[1, 2]
        elas[3, 1] = elas[1, 3]
        elas[4, 1] = elas[1, 4]
        elas[5, 1] = elas[1, 5]
        elas[3, 2] = elas[2, 3]
        elas[4, 2] = elas[2, 4]
        elas[5, 2] = elas[2, 5]
        elas[4, 3] = elas[3, 4]
        elas[5, 3] = elas[3, 5]
        elas[5, 4] = elas[4, 5]

        Title = "The triclinic crystal mechanical properties"

        self.elasproperties(Celas=elas, title=Title, theta=theta, phi=phi)

    def calc_single_modulus(self, Selas=None, theta=None, phi=None):
        """
        To calculate the single crystal modulus:
        ---
        Young's modulus

        Bulk modulus

        Shear modulus

        Poisson ratio
        """
        U = np.matrix([1, 1, 1, 0, 0, 0,])
        d = np.matrix(
            [
                [np.sin(theta) * np.cos(phi)],
                [np.sin(theta) * np.sin(phi)],
                [np.cos(theta)],
            ]
        )
        D = d * d.T
        Dv = np.matrix(
            [
                [D[0, 0]],
                [D[1, 1]],
                [D[2, 2]],
                [np.sqrt(2) * D[1, 2]],
                [np.sqrt(2) * D[0, 2]],
                [np.sqrt(2) * D[0, 1]],
            ]
        )

        # Young's modulus and Bulk modulus
        Es = 1 / (Dv.T * Selas * Dv)
        Bs = 1 / (3 * U * Selas * Dv)

        # Shear modulus
        chi = np.linspace(0, 2 * np.pi, 360)
        p_o = np.mat(np.zeros((3, 360)))
        po = np.mat(np.zeros((1, 360)))
        Go = np.mat(np.zeros((1, 360)))

        for i in np.arange(0, len(chi), 1):

            p_o[0, i] = -np.cos(theta) * np.cos(phi) * np.cos(chi[i]) + np.sin(
                phi
            ) * np.sin(chi[i])
            p_o[1, i] = -np.cos(theta) * np.sin(phi) * np.cos(chi[i]) - np.cos(
                phi
            ) * np.sin(chi[i])
            p_o[2, i] = np.sin(theta) * np.cos(chi[i])

            for j in np.arange(0, 3, 1):
                if abs(p_o[j, i]) < 1e-6:
                    p_o[j, i] = 0

            N = p_o[:, i] * p_o[:, i].T
            p_V = np.matrix(
                [
                    [N[0, 0]],
                    [N[1, 1]],
                    [N[2, 2]],
                    [np.sqrt(2) * N[1, 2]],
                    [np.sqrt(2) * N[0, 2]],
                    [np.sqrt(2) * N[0, 1]],
                ]
            )

            M = np.sqrt(2) * 0.5 * (d * p_o[:, i].T + p_o[:, i] * d.T)
            m_V = np.matrix(
                [
                    [M[0, 0]],
                    [M[1, 1]],
                    [M[2, 2]],
                    [np.sqrt(2) * M[1, 2]],
                    [np.sqrt(2) * M[0, 2]],
                    [np.sqrt(2) * M[0, 1]],
                ]
            )

            po[0, i] = -Es * Dv.T * Selas * p_V
            Go[0, i] = 1 / (2 * m_V.T * Selas * m_V)

        ps_max = po.max()
        ps_min = po.min()
        ps_avg = np.sum(po[0, :]) / len(chi)

        Gs_max = Go.max()
        Gs_min = Go.min()
        Gs_avg = np.sum(Go[0, :]) / len(chi)
        Mech = [Es[0, 0], Bs[0, 0], Gs_max, Gs_min, Gs_avg, ps_max, ps_min, ps_avg]

        return Mech
    
    def calc_Youngs_modulus(self, celas=None, theta=None, phi=None):
        """
        To calculate the single crystal modulus:
        ---
        Young's modulus

        """

        
        Selas = np.linalg.inv(celas)
        U = np.matrix([1, 1, 1, 0, 0, 0,])
        d = np.matrix(
            [
                [np.sin(theta) * np.cos(phi)],
                [np.sin(theta) * np.sin(phi)],
                [np.cos(theta)],
            ]
        )
        D = d * d.T
        Dv = np.matrix(
            [
                [D[0, 0]],
                [D[1, 1]],
                [D[2, 2]],
                [np.sqrt(2) * D[1, 2]],
                [np.sqrt(2) * D[0, 2]],
                [np.sqrt(2) * D[0, 1]],
            ]
        )

        # Young's modulus and Bulk modulus
        Es = 1/(Dv.T * (Selas * Dv))
        
        return Es[0,0]

    def calc_single_sound_elocity(self, Celas=None, theta=None, phi=None):
        """
        To calculate the single crystal sound velocity

        """
        n1 = np.sin(theta) * np.cos(phi)
        n2 = np.sin(theta) * np.sin(phi)
        n3 = np.cos(theta)

        Fv = np.zeros((3, 3))
        Fv[0, 0] = (
            n1 * n1 * Celas[0, 0]
            + n1 * n2 * Celas[0, 5]
            + n1 * n3 * Celas[0, 4]
            + n2 * n1 * Celas[5, 0]
            + n2 * n2 * Celas[5, 5]
            + n2 * n3 * Celas[5, 4]
            + n3 * n1 * Celas[4, 0]
            + n3 * n2 * Celas[4, 5]
            + n3 * n3 * Celas[4, 4]
        )

        Fv[1, 1] = (
            n1 * n1 * Celas[5, 5]
            + n1 * n2 * Celas[5, 1]
            + n1 * n3 * Celas[5, 3]
            + n2 * n1 * Celas[1, 5]
            + n2 * n2 * Celas[1, 1]
            + n2 * n3 * Celas[1, 3]
            + n3 * n1 * Celas[3, 5]
            + n3 * n2 * Celas[3, 1]
            + n3 * n3 * Celas[3, 3]
        )

        Fv[2, 2] = (
            n1 * n1 * Celas[4, 4]
            + n1 * n2 * Celas[4, 3]
            + n1 * n3 * Celas[4, 2]
            + n2 * n1 * Celas[3, 4]
            + n2 * n2 * Celas[3, 3]
            + n2 * n3 * Celas[3, 2]
            + n3 * n1 * Celas[2, 4]
            + n3 * n2 * Celas[2, 3]
            + n3 * n3 * Celas[2, 2]
        )

        Fv[0, 1] = Fv[1, 0] = (
            n1 * n1 * Celas[0, 5]
            + n1 * n2 * Celas[0, 1]
            + n1 * n3 * Celas[0, 3]
            + n2 * n1 * Celas[5, 5]
            + n2 * n2 * Celas[5, 1]
            + n2 * n3 * Celas[5, 3]
            + n3 * n1 * Celas[4, 5]
            + n3 * n2 * Celas[4, 1]
            + n3 * n3 * Celas[4, 3]
        )

        Fv[0, 2] = Fv[2, 0] = (
            n1 * n1 * Celas[0, 4]
            + n1 * n2 * Celas[0, 3]
            + n1 * n3 * Celas[0, 2]
            + n2 * n1 * Celas[5, 4]
            + n2 * n2 * Celas[5, 3]
            + n2 * n3 * Celas[5, 2]
            + n3 * n1 * Celas[4, 4]
            + n3 * n2 * Celas[4, 3]
            + n3 * n3 * Celas[4, 2]
        )

        Fv[1, 2] = Fv[2, 1] = (
            n1 * n1 * Celas[5, 4]
            + n1 * n2 * Celas[5, 3]
            + n1 * n3 * Celas[5, 2]
            + n2 * n1 * Celas[1, 4]
            + n2 * n2 * Celas[1, 3]
            + n2 * n3 * Celas[1, 2]
            + n3 * n1 * Celas[3, 4]
            + n3 * n2 * Celas[3, 3]
            + n3 * n3 * Celas[3, 2]
        )

        V_s, V_stensor = np.linalg.eigh(Fv * (10 ** 9))
        v_s = []
        for i in np.arange(0, len(V_s), 1):
            v = np.sqrt((V_s[i] / self.density)) / 1000.00
            v_s.append(v)
        return v_s

    def single_modulus_velocity(self, Celas):
        """
        Single-crystal modulus and sound velocity in 3D
        -----
        Young's modulus

        Bulk modulus  

        Shear modulus  

        Poisson's ratio  

        sound velocity
        """

        Selas = np.linalg.inv(Celas)

        # Single crystal sound velocity in 3D
        Single_V = open("single_sound_velocity_3D.out", mode="w")
        print("# theta(rad)  phi(rad)  Vs1(km/s)  Vs2(km/s)  Vl(km/s)", file=Single_V)
        for theta in np.linspace(0, np.pi / 2, 180):
            for phi in np.linspace(0, 2 * np.pi, 720):
                vs_single = self.calc_single_sound_elocity(
                    Celas=Celas, theta=theta, phi=phi
                )
                print(
                    "{:.8f}  {:.8f}   {:.4f}   {:.4f}   {:.4f}".format(
                        theta, phi, vs_single[0], vs_single[1], vs_single[2]
                    ),
                    file=Single_V,
                )
            print("", file=Single_V)

        # Single crystal modulus in 3D
        Single_modulus = open("single_modulus_3D.out", mode="w")
        print(
            "# theta(rad)  phi(rad)  Es(GPa)  Bs(GPa)  G_max(GPa)  G_min(GPa)  G_avg(GPa)  Po_max(GPa)  Po_max(GPa)  Po_max(GPa)",
            file=Single_modulus,
        )
        for theta in np.linspace(0, np.pi / 2, 90):
            for phi in np.linspace(0, 2 * np.pi, 360):
                Ms_single = self.calc_single_modulus(Selas=Selas, theta=theta, phi=phi)
                print(
                    "{:.8f}  {:.8f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(
                        theta,
                        phi,
                        Ms_single[0],
                        Ms_single[1],
                        Ms_single[2],
                        Ms_single[3],
                        Ms_single[4],
                        Ms_single[5],
                        Ms_single[6],
                        Ms_single[7],
                    ),
                    file=Single_modulus,
                )
            print("", file=Single_modulus)

    def elasproperties(self, Celas=None, title=None, theta=None, phi=None):

        Selas = np.linalg.inv(Celas)

        # Vogit
        Pv = Celas[0, 0] + Celas[1, 1] + Celas[2, 2]
        Qv = Celas[0, 1] + Celas[0, 2] + Celas[1, 2]
        Rv = Celas[3, 3] + Celas[4, 4] + Celas[5, 5]

        Ev = ((Pv + 2 * Qv) * (Pv - Qv + 3 * Rv)) / (3 * (2 * Pv + 3 * Qv + Rv))
        Gv = (Pv - Qv + 3 * Rv) / 15
        Bv = Ev * Gv / (3 * (3 * Gv - Ev))
        possion_v = (Ev / (2 * Gv)) - 1
        pwave_v = Bv + Gv * 4 / 3

        # Reuss
        Pr = Selas[0, 0] + Selas[1, 1] + Selas[2, 2]
        Qr = Selas[0, 1] + Selas[0, 2] + Selas[1, 2]
        Rr = Selas[3, 3] + Selas[4, 4] + Selas[5, 5]

        Er = 15 / (3 * Pr + 2 * Qr + Rr)
        Gr = 15 / (4 * (Pr - Qr) + 3 * Rr)
        Br = Er * Gr / (3 * (3 * Gr - Er))
        possion_r = (Er / (2 * Gr)) - 1
        pwave_r = Br + Gr * 4 / 3

        # Hill
        Eh = (Ev + Er) / 2
        Gh = (Gv + Gr) / 2
        Bh = (Bv + Br) / 2
        possion_h = (possion_v + possion_r) / 2
        pwave_h = Bh + Gh * 4 / 3

        # Polycrystal sound velocity
        Vl = np.sqrt((10 ** 9) * (Bh + 4.0 * Gh / 3.0) / self.density)
        Vs = np.sqrt((10 ** 9) * Gh / self.density)
        Vb = np.sqrt((Vl ** 2) - 4.0 * (Vs ** 2) / 3.0)
        Vm = ((1 / 3) * (1 / Vl ** 3 + 2 / Vs ** 3)) ** (-1 / 3)

        # Cauchy pressure
        Cauchy = Celas[0, 1] - Celas[3, 3]

        # Pugh'ratio
        Pugh = Gh / Bh

        # Hardness
        Hv = 2 * (Gh * Pugh ** 2) ** 0.585 - 3

        # Anisotropy index
        ## Zener anisotropy index
        A_Z = 2*Celas[3, 3] / (Celas[0, 0] - Celas[0, 1])

        ## Chung-Buessem
        A_C = (Gv - Gr) / (Gv + Gr)

        ## Universal anisotropy index
        A_U = 5 * Gv / Gr + Bv / Br - 6

        ## Log-Euclidean anisotropy
        A_L = np.sqrt(np.log(Bv / Br) ** 2 + 5 * np.log(Gv / Gr) ** 2)

        # Debye temperature
        h = 6.62607015 * 10 ** (-34)  # Plank constant
        kB = 1.380649 * 10 ** (-23)  # Boltzmann constant
        NA = 6.022140857 * 10 ** (23)  # Avogadro constant
        Debye = (
            (h / kB)
            * (
                ((0.75 * float(self.num)) / np.pi)
                * (NA * self.density * 1000 / self.mass)
            )
            ** (1 / 3)
            * Vm
        )

        # Minimum thermal conductivity
        ## Clark mode
        kmin_clark = (
            0.87
            * kB
            * (NA * float(self.num) * self.density * 1000 / self.mass) ** (2 / 3)
            * (Eh * 10 ** 9 / self.density) ** (1 / 2)
        )

        ## Chaill-Pohl model
        kmin_chaill = (
            (1 / 2.48)
            * kB
            * (float(self.num) * 10 ** 30 / self.volume) ** (2 / 3)
            * (Vl + 2 * Vs)
        )

        # Stable yes/no
        eigenvalue, eigenvector = np.linalg.eigh(Celas)
        if_stable = "Stable"
        for i in np.arange(0, 6, 1):
            if eigenvalue[i] <= 0:
                if_stable = "Unstable"

        # Output to elastic.out
        elasfile = open("second_elastic.out", mode="w")

        print(title, file=elasfile)
        if self.spg_num >= 1 and self.spg_num <= 2:
            print(
                "Triclinic crystal (spacegroup No.: {})".format(self.spg_num),
                file=elasfile,
            )

        elif self.spg_num >= 3 and self.spg_num <= 15:
            print(
                "Monoclinic crystal (spacegroup No.: {})".format(self.spg_num),
                file=elasfile,
            )

        elif self.spg_num >= 16 and self.spg_num <= 74:
            print(
                "Orthorhombic crystal (spacegroup No.: {})".format(self.spg_num),
                file=elasfile,
            )

        elif self.spg_num >= 75 and self.spg_num <= 88:
            print(
                "Tetragonal II crystal (spacegroup No.: {})".format(self.spg_num),
                file=elasfile,
            )

        elif self.spg_num >= 89 and self.spg_num <= 142:
            print(
                "Tetragonal I crystal (spacegroup No.: {})".format(self.spg_num),
                file=elasfile,
            )

        elif self.spg_num >= 143 and self.spg_num <= 148:
            print(
                "Rhombohedral II crystal (spacegroup No.: {})".format(self.spg_num),
                file=elasfile,
            )

        elif self.spg_num >= 149 and self.spg_num <= 167:
            print(
                "Rhombohedral I crystal (spacegroup No.: {})".format(self.spg_num),
                file=elasfile,
            )

        elif self.spg_num >= 168 and self.spg_num <= 194:
            print(
                "Hexagonal crystal (spacegroup No.: {})".format(self.spg_num),
                file=elasfile,
            )

        elif self.spg_num >= 195 and self.spg_num <= 230:
            print(
                "Cubic crystal (spacegroup No.: {})".format(self.spg_num),
                file=elasfile,
            )
        print("\n", end="", file=elasfile)
        print("Elastic tensor C_ij (unit: GPa)", file=elasfile)

        for i in np.arange(0, 6, 1):
            print(
                "  ",
                format(Celas[i, 0], ".3f"),
                "  ",
                format(Celas[i, 1], ".3f"),
                "  ",
                format(Celas[i, 2], ".3f"),
                "  ",
                format(Celas[i, 3], ".3f"),
                "  ",
                format(Celas[i, 4], ".3f"),
                "  ",
                format(Celas[i, 5], ".3f"),
                file=elasfile,
            )
        print("\n", end="", file=elasfile)
        print("Compliance tensor S_ij (unit: GPa^-1)", file=elasfile)

        for i in np.arange(0, 6, 1):
            print(
                "  ",
                format(Selas[i, 0], ".6f"),
                "  ",
                format(Selas[i, 1], ".6f"),
                "  ",
                format(Selas[i, 2], ".6f"),
                "  ",
                format(Selas[i, 3], ".6f"),
                "  ",
                format(Selas[i, 4], ".6f"),
                "  ",
                format(Selas[i, 5], ".6f"),
                file=elasfile,
            )

        print("\n", end="", file=elasfile)
        print("mechanical stability:  " + if_stable, file=elasfile)

        print("\n", end="", file=elasfile)

        print("unit cell volume :  {:.4f} A^3".format(self.volume), file=elasfile)
        print("unit cell density:  {:.4f} kg/m^3".format(self.density), file=elasfile)

        print("\n", end="", file=elasfile)

        print("Polycrystalline modulus", file=elasfile)
        print(
            "(Unit: GPa)",
            "Bulk modulus",
            "  ",
            "Shear modulus",
            "  ",
            "Youngs modulus",
            "  ",
            "Possion ratio",
            "  ",
            "P-wave modulus",
            file=elasfile,
        )

        print(
            "  Vogit",
            "    ",
            format(Bv, ".4f"),
            "         ",
            format(Gv, ".4f"),
            "         ",
            format(Ev, ".4f"),
            "         ",
            format(possion_v, ".4f"),
            "         ",
            format(pwave_v, ".4f"),
            file=elasfile,
        )

        print(
            "  Reuss",
            "    ",
            format(Br, ".4f"),
            "         ",
            format(Gr, ".4f"),
            "         ",
            format(Er, ".4f"),
            "         ",
            format(possion_r, ".4f"),
            "         ",
            format(pwave_r, ".4f"),
            file=elasfile,
        )

        print(
            "  Hill",
            "     ",
            format(Bh, ".4f"),
            "         ",
            format(Gh, ".4f"),
            "         ",
            format(Eh, ".4f"),
            "         ",
            format(possion_h, ".4f"),
            "         ",
            format(pwave_h, ".4f"),
            file=elasfile,
        )

        print("\n", end="", file=elasfile)

        print("Cauchy Pressure  (GPa): ", format(Cauchy, ".4f"), file=elasfile)
        print("Pugh's ratio          : ", format(Pugh, ".4f"), file=elasfile)
        print("Vickers hardness (GPa): ", format(Hv, ".4f"), file=elasfile)

        print("\n", end="", file=elasfile)

        print("Anisotropy index:", file=elasfile)
        print("  Zener anisotropy index        : ", format(A_Z, ".2f"), file=elasfile)
        print("  Chung-Buessem anisotropy index: ", format(A_C, ".2f"), file=elasfile)
        print("  Universal anisotropy index    : ", format(A_U, ".2f"), file=elasfile)
        print("  Log-Euclidean anisotropy index: ", format(A_L, ".2f"), file=elasfile)

        print("\n", end="", file=elasfile)

        print("Polycrystalline sound velocity (m/s)", file=elasfile)
        print("  Longitudinal sound velocity: ", format(Vl, ".4f"), file=elasfile)
        print("  Shear sound velocity       : ", format(Vs, ".4f"), file=elasfile)
        print("  Bulk sound velocity        : ", format(Vb, ".4f"), file=elasfile)
        print("  Average sound velocity     : ", format(Vm, ".4f"), file=elasfile)

        print("\n", end="", file=elasfile)
        # v_s = self.calc_single_sound_elocity(Celas=Celas, theta=theta, phi=phi, chi=chi)

        print("Pure single-crystal sound velocity (m/s)", file=elasfile)
        if self.spg_num >= 1 and self.spg_num <= 2:
            print(
                "  Due to the low symmetry, no corresponding information has been given yet. \
                It will be resolved in future versions.",
                file=elasfile,
            )

        elif self.spg_num >= 3 and self.spg_num <= 15:
            print(
                "  Due to the low symmetry, no corresponding information has been given yet. \
                It will be resolved in future versions.",
                file=elasfile,
            )

        elif self.spg_num >= 16 and self.spg_num <= 74:

            vl1 = np.sqrt((10 ** 9) * Celas[0, 0] / self.density)
            vs11 = np.sqrt((10 ** 9) * Celas[5, 5] / self.density)
            vs21 = np.sqrt((10 ** 9) * Celas[4, 4] / self.density)
            vl2 = np.sqrt((10 ** 9) * Celas[1, 1] / self.density)
            vs12 = np.sqrt((10 ** 9) * Celas[5, 5] / self.density)
            vs22 = np.sqrt((10 ** 9) * Celas[3, 3] / self.density)
            vl3 = np.sqrt((10 ** 9) * Celas[2, 2] / self.density)
            vs13 = np.sqrt((10 ** 9) * Celas[4, 4] / self.density)
            vs23 = np.sqrt((10 ** 9) * Celas[3, 3] / self.density)

            print(
                "  [100] direction:  vl = {:.3f}  vs1 = {:.3f}  vs2 = {:.3f}".format(
                    vl1, vs11, vs21
                ),
                file=elasfile,
            )
            print(
                "  [010] direction:  vl = {:.3f}  vs1 = {:.3f}  vs2 = {:.3f}".format(
                    vl2, vs12, vs22
                ),
                file=elasfile,
            )
            print(
                "  [001] direction:  vl = {:.3f}  vs1 = {:.3f}  vs2 = {:.3f}".format(
                    vl3, vs13, vs23
                ),
                file=elasfile,
            )

        elif self.spg_num >= 75 and self.spg_num <= 142:
            vl1 = np.sqrt((10 ** 9) * Celas[0, 0] / self.density)
            vs11 = np.sqrt((10 ** 9) * Celas[3, 3] / self.density)
            vs21 = np.sqrt((10 ** 9) * Celas[5, 5] / self.density)
            vl2 = np.sqrt((10 ** 9) * Celas[1, 1] / self.density)
            vs12 = np.sqrt((10 ** 9) * Celas[5, 5] / self.density)
            vs22 = np.sqrt((10 ** 9) * Celas[5, 5] / self.density)
            vl3 = np.sqrt(
                (10 ** 9)
                * (Celas[0, 0] + Celas[0, 1] + 2 * Celas[5, 5])
                / (2 * self.density)
            )
            vs13 = np.sqrt((10 ** 9) * Celas[3, 3] / self.density)
            vs23 = np.sqrt((10 ** 9) * (Celas[0, 0] - Celas[0, 1]) / (2 * self.density))

            print(
                "  [100] direction:  vl = {:.3f}  vs1 = {:.3f}  vs2 = {:.3f}".format(
                    vl1, vs11, vs21
                ),
                file=elasfile,
            )
            print(
                "  [001] direction:  vl = {:.3f}  vs1 = {:.3f}  vs2 = {:.3f}".format(
                    vl2, vs12, vs22
                ),
                file=elasfile,
            )
            print(
                "  [110] direction:  vl = {:.3f}  vs1 = {:.3f}  vs2 = {:.3f}".format(
                    vl3, vs13, vs23
                ),
                file=elasfile,
            )

        elif self.spg_num >= 143 and self.spg_num <= 167:
            vl1 = np.sqrt((10 ** 9) * (Celas[0, 0] - Celas[0, 1]) / (2 * self.density))
            vs11 = np.sqrt((10 ** 9) * Celas[0, 0] / self.density)
            vs21 = np.sqrt((10 ** 9) * Celas[3, 3] / self.density)
            vl2 = np.sqrt((10 ** 9) * Celas[2, 2] / self.density)
            vs12 = np.sqrt((10 ** 9) * Celas[3, 3] / self.density)
            vs22 = np.sqrt((10 ** 9) * Celas[3, 3] / self.density)

            print(
                "  [100] direction:  vl = {:.3f}  vs1 = {:.3f}  vs2 = {:.3f}".format(
                    vl1, vs11, vs21
                ),
                file=elasfile,
            )
            print(
                "  [001] direction:  vl = {:.3f}  vs1 = {:.3f}  vs2 = {:.3f}".format(
                    vl2, vs12, vs22
                ),
                file=elasfile,
            )

        elif self.spg_num >= 168 and self.spg_num <= 194:
            vl1 = np.sqrt((10 ** 9) * (Celas[0, 0] - Celas[0, 1]) / (2 * self.density))
            vs11 = np.sqrt((10 ** 9) * Celas[0, 0] / self.density)
            vs21 = np.sqrt((10 ** 9) * Celas[3, 3] / self.density)
            vl2 = np.sqrt((10 ** 9) * Celas[2, 2] / self.density)
            vs12 = np.sqrt((10 ** 9) * Celas[3, 3] / self.density)
            vs22 = np.sqrt((10 ** 9) * Celas[3, 3] / self.density)

            print(
                "  [100] direction:  vl = {:.3f}  vs1 = {:.3f}  vs2 = {:.3f}".format(
                    vl1, vs11, vs21
                ),
                file=elasfile,
            )
            print(
                "  [001] direction:  vl = {:.3f}  vs1 = {:.3f}  vs2 = {:.3f}".format(
                    vl2, vs12, vs22
                ),
                file=elasfile,
            )

        elif self.spg_num >= 195 and self.spg_num <= 230:
            vl1 = np.sqrt((10 ** 9) * Celas[0, 0] / self.density)
            vs11 = np.sqrt((10 ** 9) * Celas[3, 3] / self.density)
            vs21 = np.sqrt((10 ** 9) * Celas[3, 3] / self.density)
            vl2 = np.sqrt(
                (10 ** 9)
                * (Celas[1, 1] + Celas[0, 1] + 2 * Celas[3, 3])
                / (2 * self.density)
            )
            vs12 = np.sqrt((10 ** 9) * (Celas[0, 0] - Celas[0, 1]) / (2 * self.density))
            vs22 = np.sqrt((10 ** 9) * Celas[3, 3] / self.density)
            vl3 = np.sqrt(
                (10 ** 9)
                * (Celas[1, 1] + 2 * Celas[0, 1] + 4 * Celas[3, 3])
                / (3 * self.density)
            )
            vs13 = np.sqrt(
                (10 ** 9)
                * (Celas[1, 1] - Celas[0, 1] + Celas[3, 3])
                / (3 * self.density)
            )
            vs23 = np.sqrt(
                (10 ** 9)
                * (Celas[1, 1] - Celas[0, 1] + Celas[3, 3])
                / (3 * self.density)
            )

            print(
                "  [100] direction:  vl = {:.3f}  vs1 = {:.3f}  vs2 = {:.3f}".format(
                    vl1, vs11, vs21
                ),
                file=elasfile,
            )
            print(
                "  [110] direction:  vl = {:.3f}  vs1 = {:.3f}  vs2 = {:.3f}".format(
                    vl2, vs12, vs22
                ),
                file=elasfile,
            )
            print(
                "  [111] direction:  vl = {:.3f}  vs1 = {:.3f}  vs2 = {:.3f}".format(
                    vl3, vs13, vs23
                ),
                file=elasfile,
            )

        print("\n", end="", file=elasfile)
        
        print("Pure single-crystal Young's modulus (GPa)", file=elasfile)
        celas=Celas
        for i in np.arange(0, 3, 1):
            for j in np.arange(3, 6, 1):
                celas[i, j] = np.sqrt(2) * celas[i, j]
                celas[j, i] = np.sqrt(2) * celas[j, i]
        for m in np.arange(3, 6, 1):
            for n in np.arange(3, 6, 1):
                if m == n:
                    celas[m, n] = 2 * celas[m, n]
                else:
                    celas[m, n] = 2 * celas[m, n]
                    celas[n, m] = 2 * celas[n, m]
        E100=self.calc_Youngs_modulus(celas=celas, theta=np.pi/2,phi=0)
        E010=self.calc_Youngs_modulus(celas=celas, theta=np.pi/2,phi=np.pi/2)
        E111=self.calc_Youngs_modulus(celas=celas, theta=54*np.pi/180,phi=np.pi/4)
        E110=self.calc_Youngs_modulus(celas=celas, theta=np.pi/2,phi=np.pi/4)
        E001=self.calc_Youngs_modulus(celas=celas, theta=0,phi=0)
        print(
                "  E100 = {:.2f}  E010 = {:.2f}  E001 = {:.2f}  E110 = {:.2f}  E111 = {:.2f}".format(
                    E100, E010, E001, E110, E111
                ),
                file=elasfile,
            )
        
        print("\n", end="", file=elasfile)
        print("Debye temperature:  {:.2f} K".format(Debye), file=elasfile)

        print("\n", end="", file=elasfile)
        print("The minimum thermal conductivity (Not suitable for metallic materials):", file=elasfile)
        print("  Clark model  :  {:.3f} W/(m K)".format(kmin_clark), file=elasfile)
        print("  Chaill model :  {:.3f} W/(m K)".format(kmin_chaill), file=elasfile)


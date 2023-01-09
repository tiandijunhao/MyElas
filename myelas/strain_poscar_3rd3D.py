import numpy as np
import os
from . import read_poscar as readpos
#from shutil import copy
from . import standard_cell


class strain_poscar_3d(object):
    """
    This is used to get the strain tensor
    """

    def __init__(self):
        print("Generate strain poscar of bulk materials for static calculation.")
        self.spg_num = readpos.read_poscar().spacegroup_num()
        self.latt = standard_cell.recell(to_pricell=False).latti()
        self.atomname = readpos.read_poscar().atom_name()
        self.atomnum = standard_cell.recell(to_pricell=False).atom_number()
        self.postype = readpos.read_poscar().position_type()
        self.position = standard_cell.recell(to_pricell=False).positions()

    def __cubicI_third_elas(self, strain_max=None, strain_num=None, nelastic=None):
        starin_step = 2 * strain_max / (strain_num - 1)
        strain_param = np.arange(-strain_max, strain_max + 0.0001, starin_step)
        for i in np.arange(0, strain_num, 1):
            defmat = np.array(
                [
                    [strain_param[i], 0.0, 0.0, 0.0, 0.0, 0.0],
                    [strain_param[i], strain_param[i], 0.0, 0.0, 0.0, 0.0],
                    [strain_param[i], strain_param[i], strain_param[i], 0.0, 0.0, 0.0],
                    [strain_param[i], 0.0, 0.0, strain_param[i], 0.0, 0.0],
                    [strain_param[i], 0.0, 0.0, 0.0, 0.0, strain_param[i]],
                    [0.0, 0.0, 0.0, strain_param[i], strain_param[i], strain_param[i]],
                ]
            )
            strten = np.zeros((3, 3))
            for j in np.arange(0, nelastic, 1):
                strten[0, 0] = 2.0 * defmat[j, 0] + 1.0
                strten[0, 1] = 2.0 * defmat[j, 5]
                strten[0, 2] = 2.0 * defmat[j, 4]
                strten[1, 0] = 2.0 * defmat[j, 5]
                strten[1, 1] = 2.0 * defmat[j, 1] + 1.0
                strten[1, 2] = 2.0 * defmat[j, 3]
                strten[2, 0] = 2.0 * defmat[j, 4]
                strten[2, 1] = 2.0 * defmat[j, 3]
                strten[2, 2] = 2.0 * defmat[j, 2] + 1.0

                strten_cholesky = np.linalg.cholesky(strten)
                strten_final = 0.5 * (strten_cholesky + strten_cholesky.T)

                self.strain_latt = np.dot(self.latt, strten_final)

                self.__write_poscar(ndef=i, nelas=j)

    def __cubicII_third_elas(self, strain_max=None, strain_num=None, nelastic=None):
        starin_step = 2 * strain_max / (strain_num - 1)
        strain_param = np.arange(-strain_max, strain_max + 0.0001, starin_step)
        for i in np.arange(0, strain_num, 1):
            defmat = np.array(
                [
                    [strain_param[i], 0.0, 0.0, 0.0, 0.0, 0.0],
                    [strain_param[i], strain_param[i], 0.0, 0.0, 0.0, 0.0],
                    [strain_param[i], 0.0, strain_param[i], 0.0, 0.0, 0.0],
                    [strain_param[i], strain_param[i], strain_param[i], 0.0, 0.0, 0.0],
                    [strain_param[i], 0.0, 0.0, strain_param[i], 0.0, 0.0],
                    [strain_param[i], 0.0, 0.0, 0.0, strain_param[i], 0.0],
                    [strain_param[i], 0.0, 0.0, 0.0, 0.0, strain_param[i]],
                    [0.0, 0.0, 0.0, strain_param[i], strain_param[i], strain_param[i]],
                ]
            )
            strten = np.zeros((3, 3))
            for j in np.arange(0, nelastic, 1):
                strten[0, 0] = 2.0 * defmat[j, 0] + 1.0
                strten[0, 1] = 2.0 * defmat[j, 5]
                strten[0, 2] = 2.0 * defmat[j, 4]
                strten[1, 0] = 2.0 * defmat[j, 5]
                strten[1, 1] = 2.0 * defmat[j, 1] + 1.0
                strten[1, 2] = 2.0 * defmat[j, 3]
                strten[2, 0] = 2.0 * defmat[j, 4]
                strten[2, 1] = 2.0 * defmat[j, 3]
                strten[2, 2] = 2.0 * defmat[j, 2] + 1.0

                strten_cholesky = np.linalg.cholesky(strten)
                strten_final = 0.5 * (strten_cholesky + strten_cholesky.T)

                self.strain_latt = np.dot(self.latt, strten_final)

                self.__write_poscar(ndef=i, nelas=j)

    def __hexaI_third_elas(self, strain_max=None, strain_num=None, nelastic=None):
        starin_step = 2 * strain_max / (strain_num - 1)
        strain_param = np.arange(-strain_max, strain_max + 0.0001, starin_step)
        for i in np.arange(0, strain_num, 1):
            defmat = np.array(
                [
                    [strain_param[i], 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, strain_param[i], 0.0, 0.0, 0.0],
                    [strain_param[i], -strain_param[i], 0.0, 0.0, 0.0, 0.0],
                    [strain_param[i], 0.0, strain_param[i], 0.0, 0.0, 0.0],
                    [strain_param[i], 0.0, -strain_param[i], 0.0, 0.0, 0.0],
                    [strain_param[i], strain_param[i], 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, strain_param[i], 0.0, 0.0, strain_param[i]],
                    [strain_param[i], 0.0, 0.0, strain_param[i], 0.0, 0.0],
                    [0.0, strain_param[i], 0.0, strain_param[i], 0.0, 0.0],
                    [0.0, 0.0, strain_param[i], strain_param[i], 0.0, 0.0],
                ]
            )
            strten = np.zeros((3, 3))
            for j in np.arange(0, nelastic, 1):
                strten[0, 0] = 2.0 * defmat[j, 0] + 1.0
                strten[0, 1] = 2.0 * defmat[j, 5]
                strten[0, 2] = 2.0 * defmat[j, 4]
                strten[1, 0] = 2.0 * defmat[j, 5]
                strten[1, 1] = 2.0 * defmat[j, 1] + 1.0
                strten[1, 2] = 2.0 * defmat[j, 3]
                strten[2, 0] = 2.0 * defmat[j, 4]
                strten[2, 1] = 2.0 * defmat[j, 3]
                strten[2, 2] = 2.0 * defmat[j, 2] + 1.0

                strten_cholesky = np.linalg.cholesky(strten)
                strten_final = 0.5 * (strten_cholesky + strten_cholesky.T)

                self.strain_latt = np.dot(self.latt, strten_final)

                self.__write_poscar(ndef=i, nelas=j)

    def __hexaII_third_elas(self, strain_max=None, strain_num=None, nelastic=None):
        starin_step = 2 * strain_max / (strain_num - 1)
        strain_param = np.arange(-strain_max, strain_max + 0.0001, starin_step)
        for i in np.arange(0, strain_num, 1):
            defmat = np.array(
                [
                    [strain_param[i], 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, strain_param[i], 0.0, 0.0, 0.0],
                    [strain_param[i], -strain_param[i], 0.0, 0.0, 0.0, 0.0],
                    [strain_param[i], 0.0, strain_param[i], 0.0, 0.0, 0.0],
                    [strain_param[i], 0.0, -strain_param[i], 0.0, 0.0, 0.0],
                    [strain_param[i], strain_param[i], 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, strain_param[i], 0.0, 0.0, strain_param[i]],
                    [strain_param[i], 0.0, 0.0, strain_param[i], 0.0, 0.0],
                    [0.0, strain_param[i], 0.0, strain_param[i], 0.0, 0.0],
                    [0.0, 0.0, strain_param[i], strain_param[i], 0.0, 0.0],
                    [strain_param[i], 0.0, 0.0, 0.0, 0.0, strain_param[i]],
                    [strain_param[i], 0.0, 0.0, strain_param[i], strain_param[i], 0.0],
                ]
            )
            strten = np.zeros((3, 3))
            for j in np.arange(0, nelastic, 1):
                strten[0, 0] = 2.0 * defmat[j, 0] + 1.0
                strten[0, 1] = 2.0 * defmat[j, 5]
                strten[0, 2] = 2.0 * defmat[j, 4]
                strten[1, 0] = 2.0 * defmat[j, 5]
                strten[1, 1] = 2.0 * defmat[j, 1] + 1.0
                strten[1, 2] = 2.0 * defmat[j, 3]
                strten[2, 0] = 2.0 * defmat[j, 4]
                strten[2, 1] = 2.0 * defmat[j, 3]
                strten[2, 2] = 2.0 * defmat[j, 2] + 1.0

                strten_cholesky = np.linalg.cholesky(strten)
                strten_final = 0.5 * (strten_cholesky + strten_cholesky.T)

                self.strain_latt = np.dot(self.latt, strten_final)

                self.__write_poscar(ndef=i, nelas=j)

    def __trig_third_elas(self, strain_max=None, strain_num=None, nelastic=None):
        starin_step = 2 * strain_max / (strain_num - 1)
        strain_param = np.arange(-strain_max, strain_max + 0.0001, starin_step)
        for i in np.arange(0, strain_num, 1):
            defmat = np.array(
                [
                    [strain_param[i], 0.0, 0.0, 0.0, 0.0, 0.0],
                    [strain_param[i], strain_param[i], 0.0, 0.0, 0.0, 0.0],
                    [strain_param[i], strain_param[i], strain_param[i], 0.0, 0.0, 0.0],
                    [strain_param[i], 0.0, 0.0, strain_param[i], 0.0, 0.0],
                    [0.0, strain_param[i], 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, strain_param[i], 0.0, 0.0, 0.0],
                    [0.0, strain_param[i], strain_param[i], 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, strain_param[i], 0.0, 0.0],
                    [0.0, 0.0, strain_param[i], 0.0, 0.0, strain_param[i]],
                    [0.0, strain_param[i], 0.0, strain_param[i], 0.0, 0.0],
                    [0.0, strain_param[i], 0.0, 0.0, strain_param[i], 0.0],
                    [0.0, 0.0, strain_param[i], 0.0, strain_param[i], 0.0],
                    [strain_param[i], 0.0, strain_param[i], strain_param[i], 0.0, 0.0],
                    [strain_param[i], strain_param[i], 0.0, 0.0, strain_param[i], 0.0],
                ]
            )
            strten = np.zeros((3, 3))
            for j in np.arange(0, nelastic, 1):
                strten[0, 0] = 2.0 * defmat[j, 0] + 1.0
                strten[0, 1] = 2.0 * defmat[j, 5]
                strten[0, 2] = 2.0 * defmat[j, 4]
                strten[1, 0] = 2.0 * defmat[j, 5]
                strten[1, 1] = 2.0 * defmat[j, 1] + 1.0
                strten[1, 2] = 2.0 * defmat[j, 3]
                strten[2, 0] = 2.0 * defmat[j, 4]
                strten[2, 1] = 2.0 * defmat[j, 3]
                strten[2, 2] = 2.0 * defmat[j, 2] + 1.0

                strten_cholesky = np.linalg.cholesky(strten)
                strten_final = 0.5 * (strten_cholesky + strten_cholesky.T)

                self.strain_latt = np.dot(self.latt, strten_final)

                self.__write_poscar(ndef=i, nelas=j)

    def __tetra_third_elas(self, strain_max=None, strain_num=None, nelastic=None):
        starin_step = 2 * strain_max / (strain_num - 1)
        strain_param = np.arange(-strain_max, strain_max + 0.0001, starin_step)
        for i in np.arange(0, strain_num, 1):
            defmat = np.array(
                [
                    [strain_param[i], 0.0, 0.0, 0.0, 0.0, 0.0],  # A1
                    [strain_param[i], strain_param[i], 0.0, 0.0, 0.0, 0.0],  # A2
                    [
                        strain_param[i],
                        strain_param[i],
                        strain_param[i],
                        0.0,
                        0.0,
                        0.0,
                    ],  # A3
                    [strain_param[i], 0.0, 0.0, strain_param[i], 0.0, 0.0],  # A4
                    [strain_param[i], 0.0, 0.0, 0.0, 0.0, strain_param[i]],  # A5
                    [strain_param[i], 0.0, strain_param[i], 0.0, 0.0, 0.0],  # A6
                    # [0.0, strain_param[i], 0.0, 0.0, 0.0, 0.0], #A7
                    [0.0, 0.0, strain_param[i], 0.0, 0.0, 0.0],  # A8
                    [0.0, strain_param[i], strain_param[i], 0.0, 0.0, 0.0],  # A9
                    # [strain_param[i] / 3.0, strain_param[i], 0.0, 0.0, 0.0, 0.0], #A10
                    [0.0, 0.0, strain_param[i], 0.0, 0.0, strain_param[i]],  # A11
                    # [0.0, strain_param[i], 0.0, strain_param[i], 0.0, 0.0], #A12
                    # [0.0, strain_param[i], 0.0, 0.0, strain_param[i], 0.0], #A13
                    # [0.0, 0.0, strain_param[i], 0.0, strain_param[i], 0.0], #A14
                    # [0.0, strain_param[i] / 3.0, strain_param[i], 0.0, 0.0, 0.0], #A15
                    [strain_param[i] / 3.0, strain_param[i], 0.0, 0.0, 0.0, 0.0],  # A16
                    [strain_param[i], 0.0, 0.0, 0.0, strain_param[i], 0.0],  # A17
                    # [0.0, strain_param[i], 0.0, 0.0, 0.0, strain_param[i]], #A18
                    [0.0, 0.0, strain_param[i], strain_param[i], 0.0, 0.0],  # A19
                    # [strain_param[i], 0.0, strain_param[i], 0.0, 0.0, 0.0], #A20
                ]
            )
            strten = np.zeros((3, 3))
            for j in np.arange(0, nelastic, 1):
                strten[0, 0] = 2.0 * defmat[j, 0] + 1.0
                strten[0, 1] = 2.0 * defmat[j, 5]
                strten[0, 2] = 2.0 * defmat[j, 4]
                strten[1, 0] = 2.0 * defmat[j, 5]
                strten[1, 1] = 2.0 * defmat[j, 1] + 1.0
                strten[1, 2] = 2.0 * defmat[j, 3]
                strten[2, 0] = 2.0 * defmat[j, 4]
                strten[2, 1] = 2.0 * defmat[j, 3]
                strten[2, 2] = 2.0 * defmat[j, 2] + 1.0

                strten_cholesky = np.linalg.cholesky(strten)
                strten_final = 0.5 * (strten_cholesky + strten_cholesky.T)

                self.strain_latt = np.dot(self.latt, strten_final)

                self.__write_poscar(ndef=i, nelas=j)

    def __orth_third_elas(self, strain_max=None, strain_num=None, nelastic=None):
        starin_step = 2 * strain_max / (strain_num - 1)
        strain_param = np.arange(-strain_max, strain_max + 0.0001, starin_step)
        for i in np.arange(0, strain_num, 1):
            defmat = np.array(
                [
                    [strain_param[i], 0.0, 0.0, 0.0, 0.0, 0.0],  # A1
                    [strain_param[i], strain_param[i], 0.0, 0.0, 0.0, 0.0],  # A2
                    [
                        strain_param[i],
                        strain_param[i],
                        strain_param[i],
                        0.0,
                        0.0,
                        0.0,
                    ],  # A3
                    [strain_param[i], 0.0, 0.0, strain_param[i], 0.0, 0.0],  # A4
                    [strain_param[i], 0.0, 0.0, 0.0, 0.0, strain_param[i]],  # A5
                    [strain_param[i], 0.0, strain_param[i], 0.0, 0.0, 0.0],  # A6
                    [0.0, strain_param[i], 0.0, 0.0, 0.0, 0.0],  # A7
                    [0.0, 0.0, strain_param[i], 0.0, 0.0, 0.0],  # A8
                    [0.0, strain_param[i], strain_param[i], 0.0, 0.0, 0.0],  # A9
                    [strain_param[i] / 3.0, strain_param[i], 0.0, 0.0, 0.0, 0.0],  # A10
                    [0.0, 0.0, strain_param[i], 0.0, 0.0, strain_param[i]],  # A11
                    [0.0, strain_param[i], 0.0, strain_param[i], 0.0, 0.0],  # A12
                    [0.0, strain_param[i], 0.0, 0.0, strain_param[i], 0.0],  # A13
                    [0.0, 0.0, strain_param[i], 0.0, strain_param[i], 0.0],  # A14
                    [0.0, strain_param[i] / 3.0, strain_param[i], 0.0, 0.0, 0.0],  # A15
                    [strain_param[i] / 3.0, strain_param[i], 0.0, 0.0, 0.0, 0.0],  # A16
                    [strain_param[i], 0.0, 0.0, 0.0, strain_param[i], 0.0],  # A17
                    [0.0, strain_param[i], 0.0, 0.0, 0.0, strain_param[i]],  # A18
                    [0.0, 0.0, strain_param[i], strain_param[i], 0.0, 0.0],  # A19
                    [strain_param[i], 0.0, strain_param[i], 0.0, 0.0, 0.0],  # A20
                ]
            )
            strten = np.zeros((3, 3))
            for j in np.arange(0, nelastic, 1):
                strten[0, 0] = 2.0 * defmat[j, 0] + 1.0
                strten[0, 1] = 2.0 * defmat[j, 5]
                strten[0, 2] = 2.0 * defmat[j, 4]
                strten[1, 0] = 2.0 * defmat[j, 5]
                strten[1, 1] = 2.0 * defmat[j, 1] + 1.0
                strten[1, 2] = 2.0 * defmat[j, 3]
                strten[2, 0] = 2.0 * defmat[j, 4]
                strten[2, 1] = 2.0 * defmat[j, 3]
                strten[2, 2] = 2.0 * defmat[j, 2] + 1.0

                strten_cholesky = np.linalg.cholesky(strten)
                strten_final = 0.5 * (strten_cholesky + strten_cholesky.T)

                self.strain_latt = np.dot(self.latt, strten_final)

                self.__write_poscar(ndef=i, nelas=j)

    def __write_poscar(self, ndef=None, nelas=None):
        """
        To write the strain POSCAR.
        """

        full_path = (
            "thirdnelastic_"
            + str(format(nelas + 1, "02d"))
            + "/strain_"
            + str(format(ndef + 1, "03d"))
        )

        if os.path.exists(full_path):
            pass
        else:
            os.makedirs(full_path)

        # writepos = open(
        #    "POSCAR_"
        #    + str(format(nelas + 1, "02d"))
        #    + "_"
        #    + str(format(ndef + 1, "03d")),
        #    mode="w",
        # )
        writepos = open(full_path + "/POSCAR", mode="w",)
        print("strain_poscar", file=writepos)
        print("1.0", file=writepos)
        for m in np.arange(0, 3, 1):
            print(
                format(self.strain_latt[m, 0], ".10f"),
                "   ",
                format(self.strain_latt[m, 1], ".10f"),
                "   ",
                format(self.strain_latt[m, 2], ".10f"),
                file=writepos,
            )

        for j in np.arange(0, len(self.atomname), 1):
            print(self.atomname[j], file=writepos, end=" ")
        print(end="\n", file=writepos)
        for l in np.arange(0, len(self.atomname), 1):
            print(int(self.atomnum[l]), end=" ", file=writepos)
        print(end="\n", file=writepos)
        print(self.postype[0], file=writepos)

        for n in np.arange(0, self.position.shape[0], 1):
            print(
                format(self.position[n, 0], ".10f"),
                "   ",
                format(self.position[n, 1], ".10f"),
                "   ",
                format(self.position[n, 2], ".10f"),
                file=writepos,
            )
        writepos.close()

        # if os.path.isfile("KPOINTS"):
        #    copy("./KPOINTS", full_path)
        # else:
        #    pass

        #
        # if os.path.isfile("POTCAR"):
        #    copy("./POTCAR", full_path)
        # else:
        #    pass

    def crystal_strain(self, strainmax=None, strainnum=None):
        """
        According the crystal to produce the strain poscar
        strainmax: The maxinum of the strain, must be positive. eg: 0.018
        strainnum: The number of strain. eg: 13
        """

        # if self.spg_num >= 1 and self.spg_num <= 2:
        #    self.__triclinic_strain(
        #        strain_max=strainmax, strain_num=strainnum, nelastic=21
        #    )
        #
        # elif self.spg_num >= 3 and self.spg_num <= 15:
        #    self.__monoclinic_strain(
        #        strain_max=strainmax, strain_num=strainnum, nelastic=13
        #    )

        if self.spg_num >= 16 and self.spg_num <= 74:
            self.__orth_third_elas(
                strain_max=strainmax, strain_num=strainnum, nelastic=20
            )

        elif self.spg_num >= 75 and self.spg_num <= 142:
            self.__tetra_third_elas(
                strain_max=strainmax, strain_num=strainnum, nelastic=12
            )

        # elif self.spg_num >= 143 and self.spg_num <= 148:
        #    self.__rhombohedral_II_strain(
        #        strain_max=strainmax, strain_num=strainnum, nelastic=8
        #    )

        elif self.spg_num >= 149 and self.spg_num <= 167:
            self.__trig_third_elas(
                strain_max=strainmax, strain_num=strainnum, nelastic=14
            )

        elif self.spg_num >= 168 and self.spg_num <= 176:
            self.__hexaII_third_elas(
                strain_max=strainmax, strain_num=strainnum, nelastic=12
            )

        elif self.spg_num >= 177 and self.spg_num <= 194:
            self.__hexaI_third_elas(
                strain_max=strainmax, strain_num=strainnum, nelastic=10
            )

        elif self.spg_num >= 195 and self.spg_num <= 206:
            self.__cubicII_third_elas(
                strain_max=strainmax, strain_num=strainnum, nelastic=8
            )

        elif self.spg_num >= 207 and self.spg_num <= 230:
            self.__cubicI_third_elas(
                strain_max=strainmax, strain_num=strainnum, nelastic=6
            )


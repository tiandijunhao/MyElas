import numpy as np
import os
from . import read_poscar as readpos
#from shutil import copy
from . import standard_cell


class strain_poscar_2d(object):
    """
    This is used to get the strain tensor
    """

    def __init__(
        self,
        spg_num=None,
        latt=None,
        atomname=None,
        atomnum=None,
        postype=None,
        position=None,
        strain_latt=None,
    ):
        self.spg_num = readpos.read_poscar().spacegroup_num()
        self.latt = readpos.read_poscar().latti()
        self.atomname = readpos.read_poscar().atom_name()
        self.atomnum = readpos.read_poscar().atom_number()
        self.postype = readpos.read_poscar().position_type()
        self.position = readpos.read_poscar().positions()

    def strain_2D(self, strain_max=None, strain_num=None):
        """
        Strain matrix of cubic system

        strain_max: The maximum of strain, must be positive number

        strain_num: The number of strain, must be positive number

        nelastic: Independent elastic component

        """
        if self.spg_num>=3 and self.spg_num<=15:
            nelastic = 6
        else:
            nelastic = 4

        starin_step = 2 * strain_max / (strain_num - 1)
        strain_param = np.arange(-strain_max, strain_max + 0.0001, starin_step)
        for i in np.arange(0, strain_num, 1):
            if self.spg_num>=3 and self.spg_num<=15:
                defmat = np.array(
                    [
                        [strain_param[i], 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, strain_param[i], 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, strain_param[i]],
                        [strain_param[i], strain_param[i], 0.0, 0.0, 0.0, 0.0],
                        [strain_param[i], 0.0, 0.0, 0.0, 0.0, strain_param[i]],
                        [0.0, strain_param[i], 0.0, 0.0, 0.0, strain_param[i]],
                    ]
                )
            
            elif self.spg_num <3:
                print('Error! The Space group is wrong!')
            
            else:
                defmat = np.array(
                    [
                        [strain_param[i], 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, strain_param[i], 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, strain_param[i]],
                        [strain_param[i], strain_param[i], 0.0, 0.0, 0.0, 0.0],
                    ]
                )
            
            strten = np.zeros((3, 3))
            for j in np.arange(0, nelastic, 1):
                strten[0, 0] = defmat[j, 0] + 1.0
                strten[0, 1] = 0.5 * defmat[j, 5]
                strten[0, 2] = 0.5 * defmat[j, 4]
                strten[1, 0] = 0.5 * defmat[j, 5]
                strten[1, 1] = defmat[j, 1] + 1.0
                strten[1, 2] = 0.5 * defmat[j, 3]
                strten[2, 0] = 0.5 * defmat[j, 4]
                strten[2, 1] = 0.5 * defmat[j, 3]
                strten[2, 2] = defmat[j, 2] + 1.0

                self.strain_latt = np.dot(self.latt, strten)

                self.__write_poscar(ndef=i, nelas=j)

    def __write_poscar(self, ndef=None, nelas=None):
        """
        To write the strain POSCAR.
        """

        full_path = (
            "2D_nelastic_"
            + str(format(nelas + 1, "02d"))
            + "/strain_"
            + str(format(ndef + 1, "03d"))
        )

        if os.path.exists(full_path):
            pass
        else:
            os.makedirs(full_path)

        # writepos = open("POSCAR_"+str(format(nelas + 1, "02d")) +
        #                "_"+str(format(ndef + 1, "03d")), mode="w",)
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

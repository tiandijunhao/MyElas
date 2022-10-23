import numpy as np
import os
from . import read_poscar as readpos
from . import strain_poscar_2D as strain2d
from . import strain_poscar_3D as strain3d
from . import strain_poscar_3rd3D as strain3rd3d

# from shutil import copy
from . import standard_cell


class parameter_test(object):
    """
    Generate vasp file
    """

    def __init__(self):
        print("Generate a VASP input file!")

    def INCAR_G(
        self,
        encut=None,
        kspace=None,
        ismear=0,
        sigma=0.05,
        isif=2,
        pressure=None,
        Temperature=None,
        Calc_type=None,
    ):
        if Calc_type == "rlx":
            incar_rlx = open("INCAR", mode="w")

            print("Global Parameters", file=incar_rlx)
            print(" ISTART =  0       ", file=incar_rlx)
            print(" ICHARG =  2       ", file=incar_rlx)
            print(" LREAL  = F        ", file=incar_rlx)
            print(" PREC   =  Accurate", file=incar_rlx)
            print(" LWAVE  = .FALSE.  ", file=incar_rlx)
            print(" LCHARG = .FALSE.  ", file=incar_rlx)
            print(" ADDGRID= .TRUE.   ", file=incar_rlx)
            if encut <= 0 or encut == None:
                print(" ENCUT = 500", file=incar_rlx)
            else:
                print(" ENCUT = " + str(encut), file=incar_rlx)
            print("\n", file=incar_rlx)

            print("Electronic Relaxation", file=incar_rlx)
            if ismear != 0:
                print(" ISMEAR = " + str(ismear), file=incar_rlx)
            else:
                print(" ISMEAR = 0", file=incar_rlx)

            if sigma <= 0.0:
                print(" SIGMA  = 0.05", file=incar_rlx)
            else:
                print(" SIGMA  = " + str(sigma), file=incar_rlx)
            print(" NELM   =  40      ", file=incar_rlx)
            print(" NELMIN =  4       ", file=incar_rlx)
            print(" EDIFF  =  1E-06   ", file=incar_rlx)
            print(" GGA  =  PE        ", file=incar_rlx)
            print("\n", file=incar_rlx)

            print("Ionic Relaxation", file=incar_rlx)
            print(" NELMIN =  6       ", file=incar_rlx)
            print(" NSW    =  40     ", file=incar_rlx)
            print(" IBRION =  2       ", file=incar_rlx)
            if isif >= 0 and isif <= 7:
                print(" ISIF = " + str(isif), file=incar_rlx)
            else:
                print(" ISIF = 2", file=incar_rlx)
            print(" EDIFFG = -0.001    ", file=incar_rlx)
            print(" # ISYM =  2       ", file=incar_rlx)
            print("\n", file=incar_rlx)

            print("Other parameters", file=incar_rlx)
            if pressure == 0:
                print(" # PSTRESS=", file=incar_rlx)
            else:
                print(" PSTRESS= {:.2f}".format(pressure * 10.0), file=incar_rlx)

            if kspace <= 0.0:
                print(" #KSPACING=None", file=incar_rlx)
            else:
                print(" KSPACING=" + str(kspace), file=incar_rlx)
            print(" #KGAMMA=.FALSE.", file=incar_rlx)
            print(" #KPAR=4", file=incar_rlx)

        elif Calc_type == "stc":
            incar_stc = open("INCAR", mode="w")

            print("Global Parameters", file=incar_stc)
            print(" ISTART =  0       ", file=incar_stc)
            print(" ICHARG =  2       ", file=incar_stc)
            print(" LREAL  = F        ", file=incar_stc)
            print(" PREC   =  Accurate", file=incar_stc)
            print(" LWAVE  = .FALSE.  ", file=incar_stc)
            print(" LCHARG = .FALSE.  ", file=incar_stc)
            print(" ADDGRID= .TRUE.   ", file=incar_stc)
            if encut <= 0 or encut == None:
                print(" ENCUT = 500", file=incar_stc)
            else:
                print(" ENCUT = " + str(encut), file=incar_stc)
            print("\n", file=incar_stc)

            print("Electronic Relaxation", file=incar_stc)
            if Temperature <= 0:
                print(" ISMEAR = " + str(ismear), file=incar_stc)
                if ismear == -5:
                    print(" #SIGMA=" + str(sigma), file=incar_stc)
                else:
                    print(" SIGMA = " + str(sigma), file=incar_stc)
            else:
                print(" ISMEAR = -1", file=incar_stc)
                kBT = 8.6173333 * 10 ** (-5) * Temperature
                print(" SIGMA = {:.6f}".format(kBT), file=incar_stc)
            print(" #NELM   =  40      ", file=incar_stc)
            print(" #NELMIN =  4       ", file=incar_stc)
            print(" EDIFF  =  1E-06   ", file=incar_stc)
            print(" GGA  =  PE        ", file=incar_stc)
            print("\n", file=incar_stc)

            print("Ionic Relaxation", file=incar_stc)
            print(" #NELMIN =  6       ", file=incar_stc)
            print(" #NSW    =  40     ", file=incar_stc)
            print(" #IBRION =  2       ", file=incar_stc)
            print(" ISIF = 2           ", file=incar_stc)
            print(" EDIFFG = -0.001    ", file=incar_stc)
            print(" # ISYM =  2       ", file=incar_stc)
            print("\n", file=incar_stc)

            print("Other parameters", file=incar_stc)
            if pressure == 0:
                print(" # PSTRESS=", file=incar_stc)
            else:
                print(" PSTRESS = {:.2f}".format(pressure * 10.0), file=incar_stc)

            if kspace <= 0.0:
                print(" #KSPACING=None", file=incar_stc)
            else:
                print(" KSPACING=" + str(kspace), file=incar_stc)
            print(" #KGAMMA=.FALSE.", file=incar_stc)
            print(" #KPAR=4", file=incar_stc)

    def KPOINTS_G(self, kpoint=None, mgrid=None):

        kpoints_file = open("KPOINTS", mode="w")

        print("A", file=kpoints_file)
        print("0", file=kpoints_file)
        if mgrid == None:
            print("Gamma", file=kpoints_file)
        else:
            print(mgrid, file=kpoints_file)
        if kpoint > 0:
            print("{:d} {:d} {:d}".format(kpoint, kpoint, kpoint), file=kpoints_file)
        else:
            print("Please set KPOINT >0")
            print("8 8 8", file=kpoints_file)
        print("0 0 0", file=kpoints_file)

    def KPOINTS_G_2D(self, kpoint=None, mgrid=None):

        kpoints_file = open("KPOINTS", mode="w")

        print("A", file=kpoints_file)
        print("0", file=kpoints_file)
        if mgrid == None:
            print("Gamma", file=kpoints_file)
        else:
            print(mgrid, file=kpoints_file)
        print("{:d} {:d} 1".format(kpoint, kpoint), file=kpoints_file)
        print("0 0 0", file=kpoints_file)

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
        ediff=1e-6,
        ediffg=-0.001,
        pressure=None,
        eTemperature=0,
        Temperature=None,
        ml = 0,
        mistart = 1,
        nsw = 10000,
        potim = 1,
        lgamma =20.0,
        lgammal=20.0,
        pmass = 20.0,
        Calc_type=None,
    ):
        if Calc_type == "rlx":
            print("Generate INCAR for relax calculation.")
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
            print(" EDIFF  = " +str(ediff), file=incar_rlx)
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
            print(" EDIFFG  = " +str(ediffg), file=incar_rlx)
            print(" # ISYM =  2       ", file=incar_rlx)
            print("\n", file=incar_rlx)

            print("Other parameters", file=incar_rlx)
            if pressure == 0:
                print(" # PSTRESS=", file=incar_rlx)
            else:
                print(" PSTRESS= {:.2f}".format(pressure * 10.0), file=incar_rlx)

            if kspace <= 0.0 or kspace == None:
                print(" #KSPACING=None", file=incar_rlx)
            else:
                print(" KSPACING=" + str(kspace), file=incar_rlx)
            print(" #KGAMMA=.FALSE.", file=incar_rlx)
            print(" #KPAR=4", file=incar_rlx)

        elif Calc_type == "stc":
            print("Generate INCAR for static calculation.")
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
            if eTemperature <= 0:
                print(" ISMEAR = " + str(ismear), file=incar_stc)
                if ismear == -5:
                    print(" #SIGMA=" + str(sigma), file=incar_stc)
                else:
                    print(" SIGMA = " + str(sigma), file=incar_stc)
            else:
                print(" ISMEAR = -1", file=incar_stc)
                kBT = 8.6173333 * 10 ** (-5) * eTemperature
                print(" SIGMA = {:.6f}".format(kBT), file=incar_stc)
            print(" #NELM   =  40      ", file=incar_stc)
            print(" #NELMIN =  4       ", file=incar_stc)
            print(" EDIFF  = " +str(ediff), file=incar_stc)
            print(" GGA  =  PE        ", file=incar_stc)
            print("\n", file=incar_stc)

            print("Ionic Relaxation", file=incar_stc)
            print(" #NELMIN =  6       ", file=incar_stc)
            print(" #NSW    =  40     ", file=incar_stc)
            print(" #IBRION =  2       ", file=incar_stc)
            print(" ISIF = 2           ", file=incar_stc)
            print(" EDIFFG  = " +str(ediffg), file=incar_stc)
            print(" # ISYM =  2       ", file=incar_stc)
            print("\n", file=incar_stc)

            print("Other parameters", file=incar_stc)
            if pressure == 0:
                print(" # PSTRESS=", file=incar_stc)
            else:
                print(" PSTRESS = {:.2f}".format(pressure * 10.0), file=incar_stc)

            if kspace <= 0.0 or kspace ==None:
                print(" #KSPACING=None", file=incar_stc)
            else:
                print(" KSPACING=" + str(kspace), file=incar_stc)
            print(" #KGAMMA=.FALSE.", file=incar_stc)
            print(" #KPAR=4", file=incar_stc)
        
        elif Calc_type == "nvt":
            print("Generate INCAR for NVT MD calculation.")
            incar_nvt = open("INCAR_NVT", mode="w")

            print("Global Parameters    ", file=incar_nvt)  
            print("  ISTART =  0        ", file=incar_nvt)  
            print("  LREAL  =  Auto     ", file=incar_nvt)  
            if encut <= 0 or encut == None:
                print("  ENCUT = 500", file=incar_nvt)
            else:
                print("  ENCUT = " + str(encut), file=incar_nvt)  
            print("  PREC   =  Normal   ", file=incar_nvt) 
            print("  LWAVE  = .FALSE.   ", file=incar_nvt)  
            print("  LCHARG = .FALSE.   ", file=incar_nvt)  
            print("  GGA = PE           ", file=incar_nvt)    
            print("  #NCORE= 2          ", file=incar_nvt) 
            print("  #KPAR = 4          ", file=incar_nvt)   
            print("  #KSPACING = 0.25   ", file=incar_nvt)     
            print("  #KGAMMA = .FALSE.  ", file=incar_nvt)      
            if pressure == 0:
                print("  # PSTRESS=", file=incar_nvt)
            else:
                print("  PSTRESS= {:.2f}".format(pressure * 10.0), file=incar_nvt)       
            print("                     ", file=incar_nvt)   
            print("Electronic Relaxation", file=incar_nvt)  
            if eTemperature <= 0 or eTemperature==None:
                print("  ISMEAR = " + str(ismear), file=incar_nvt)
                if ismear == -5:
                    print("  #SIGMA=" + str(sigma), file=incar_nvt)
                else:
                    print("  SIGMA = " + str(sigma), file=incar_nvt)
            else:
                print("  ISMEAR = -1", file=incar_nvt)
                kBT = 8.6173333 * 10 ** (-5) * eTemperature
                print("  SIGMA = {:.6f}".format(kBT), file=incar_nvt)     
            print("  EDIFF  = "+str(ediff), file=incar_nvt)      
            print("  ALGO = Normal      ", file=incar_nvt)    
            print("  NELMIN = 4         ", file=incar_nvt)        
            print("                     ", file=incar_nvt) 
            print("Molecular Dynamics   ", file=incar_nvt)    
            print("  IBRION =  0        ", file=incar_nvt)        
            print("  NSW    = "+str(nsw), file=incar_nvt)              
            print("  POTIM  = "+str(potim), file=incar_nvt)        
            print("  SMASS  =  0        ", file=incar_nvt)
            if Temperature == None:
                print("Please set Temperature in MD.")       
            print("  TEBEG  = "+str(Temperature), file=incar_nvt)        
            print("  TEEND  = "+str(Temperature), file=incar_nvt)        
            print("  MDALGO =  2        ", file=incar_nvt)        
            print("  ISYM   =  0        ", file=incar_nvt)        
            print("  ISIF = 2           ", file=incar_nvt)  
            print("                     ", file=incar_nvt)
            print("Machine learning     ", file=incar_nvt)      
            if ml == 1:      
                print("  ML_LMLFF = .TRUE.  ", file=incar_nvt)  
            else:
                print("  ML_LMLFF = .FALSE.  ", file=incar_nvt)   
            print("  ML_ISTART = "+str(mistart), file=incar_nvt)   
            print("  ML_RCUT1 = 6.0     ", file=incar_nvt) 
            print("  ML_RCUT2 = 6.0     ", file=incar_nvt)     
            print("  ML_MB = 3000       ", file=incar_nvt) 
            print("  ML_MCONF = 3000    ", file=incar_nvt)  
            
        elif Calc_type == "npt":
            print("Generate INCAR for NPT MD calculation.")
            print("Please note LANGEVIN_GAMMA parameter in INCAR.")
            incar_npt = open("INCAR_NPT", mode="w")

            print("Global Parameters    ", file=incar_npt)  
            print("  ISTART =  0        ", file=incar_npt)  
            print("  LREAL  =  Auto     ", file=incar_npt)  
            if encut <= 0 or encut == None:
                print("  ENCUT = 500", file=incar_npt)
            else:
                print("  ENCUT = " + str(encut), file=incar_npt)
            print("  PREC   =  Normal   ", file=incar_npt) 
            print("  LWAVE  = .FALSE.   ", file=incar_npt)  
            print("  LCHARG = .FALSE.   ", file=incar_npt)  
            print("  GGA = PE           ", file=incar_npt)    
            print("  #NCORE= 2          ", file=incar_npt) 
            print("  #KPAR = 4          ", file=incar_npt)   
            print("  #KSPACING = 0.25   ", file=incar_npt)     
            print("  #KGAMMA = .FALSE.  ", file=incar_npt)      
            if pressure == 0:
                print("  # PSTRESS=", file=incar_npt)
            else:
                print("  PSTRESS= {:.2f}".format(pressure * 10.0), file=incar_npt)       
            print("                     ", file=incar_npt)   
            print("Electronic Relaxation", file=incar_npt)  
            if eTemperature <= 0 or eTemperature==None:
                print("  ISMEAR = " + str(ismear), file=incar_npt)
                if ismear == -5:
                    print("  #SIGMA=" + str(sigma), file=incar_npt)
                else:
                    print("  SIGMA = " + str(sigma), file=incar_npt)
            else:
                print("  ISMEAR = -1", file=incar_npt)
                kBT = 8.6173333 * 10 ** (-5) * eTemperature
                print("  SIGMA = {:.6f}".format(kBT), file=incar_npt)    
            print("  EDIFF  = "+str(ediff), file=incar_npt)       
            print("  ALGO = Normal      ", file=incar_npt)    
            print("  NELMIN = 4         ", file=incar_npt)        
            print("                     ", file=incar_npt) 
            print("Molecular Dynamics   ", file=incar_npt)    
            print("  IBRION =  0        ", file=incar_npt)        
            print("  NSW    = "+str(nsw), file=incar_npt)              
            print("  POTIM  = "+str(potim), file=incar_npt)        
            print("  SMASS  =  0        ", file=incar_npt)
            if Temperature == None:
                print("Please set Temperature in MD.")       
            print("  TEBEG  = "+str(Temperature), file=incar_npt)        
            print("  TEEND  = "+str(Temperature), file=incar_npt)        
            print("  MDALGO =  3        ", file=incar_npt)        
            print("  ISYM   =  0        ", file=incar_npt)        
            print("  ISIF = 3           ", file=incar_npt)
            print("                     ", file=incar_npt)
            print("  LANGEVIN_GAMMA= ", end='',file=incar_npt)
            n = len(lgamma)
            for i in np.arange(0,n):
                print("{} ".format(str(lgamma[i])), end='', file=incar_npt)
            print(" ", file=incar_npt)
            print("  LANGEVIN_GAMMA_L= "+str(lgammal), file=incar_npt)     
            print("  PMASS= "+str(pmass), file=incar_npt)   
            print("                     ", file=incar_npt)
            print("Machine learning     ", file=incar_npt)
            if ml == 1:      
                print("  ML_LMLFF = .TRUE.  ", file=incar_npt)  
            else:
                print("  ML_LMLFF = .FALSE.  ", file=incar_npt)   
            print("  ML_ISTART = "+str(mistart), file=incar_npt)   
            print("  ML_RCUT1 = 6.0     ", file=incar_npt) 
            print("  ML_RCUT2 = 6.0     ", file=incar_npt)     
            print("  ML_MB = 3000       ", file=incar_npt) 
            print("  ML_MCONF = 3000    ", file=incar_npt) 

    def KPOINTS_G(self, kpoint=None, mgrid=None):
        print("Generate KPOINTS for bulk materials.")
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
        print("Generate KPOINTS for 2D materials.")
        kpoints_file = open("KPOINTS", mode="w")

        print("A", file=kpoints_file)
        print("0", file=kpoints_file)
        if mgrid == None:
            print("Gamma", file=kpoints_file)
        else:
            print(mgrid, file=kpoints_file)
        print("{:d} {:d} 1".format(kpoint, kpoint), file=kpoints_file)
        print("0 0 0", file=kpoints_file)

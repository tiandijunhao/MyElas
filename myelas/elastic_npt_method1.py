import numpy as np
import shutil,os
from . import read_poscar as readpos
from . import standard_cell


class npt_solve_method_1(object):
    def __init__(self, Temp):
        print("==========================================================================")
        print("==== Stress-strain method in constant pressure ensembles               ===")
        print("==== This method was proposed by Yunguo Li and used in ElasT software. ===")
        print("==== We have made some improvements compared to the ElasT.             ===")
        print("==== Cite: Comput. Phys. Commun., 273 (2022) 108280.                   ===")
        print("==========================================================================")
        self.Temp = Temp
        self.spg_num = readpos.read_poscar().spacegroup_num()
        self.lattindex = readpos.read_poscar().latt_index()
        self.latt = standard_cell.recell(to_pricell=False).latti()
        atom_num = readpos.read_poscar().atom_number()
        
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
            self.num = self.num + int(atom_num[i])
        #print("Total mass: ", self.mass)

    def read_sxdatcar(self, sstep=None, slice_step=None):
        lattice_avg = np.zeros((3, 3), dtype=np.float64)
        
        print("Start reading the lattice matrix and stress tensor.")
        xdatacar = open("XDATCAR", mode="r")
        if os.path.isfile("stress.out"):
            pass
        else:
            if os.path.isfile("OUTCAR"):
                os.system("grep 'Total+kin' OUTCAR |awk '{print $2,$3,$4,$5,$6,$7}' >stress.out")
            else:
                print("Warning: No stress.out or OUTCAR")
        stress_data = np.loadtxt("stress.out")
        stress = np.zeros((len(stress_data[:,0]), 6), dtype=np.float64)
        all_lines = xdatacar.readlines()
        inum = 0
        tstep = 0
        
        for i in np.arange(0,6):
            stress[:,i]=stress_data[:,i]-np.mean(stress_data[:,i])
        
        for line in np.arange(0, len(all_lines), 1):
            if "Direct configuration=" in all_lines[line]:
                tstep = tstep + 1
        print("XDATCAR total step: {}; Start calculate step: {}".format(tstep, sstep))
        print("Stress total step: {}; Start calculate step: {}".format(len(stress_data[:,0]), sstep))
        if tstep != len(stress_data[:,0]):
                print("Error! stress numbers != strain numbers")
        index_num = tstep - sstep
        if slice_step>index_num:
            print("Slice step is too large! Now it equals to (total_step-input_start_step)")
            slice_step = index_num
        n_num = int(index_num /slice_step)
        lattice_i = np.zeros((index_num, 3, 3), dtype=np.float64)
        stress_tensor = np.zeros((index_num,6))
        x_num = 0
        for line in np.arange(0, len(all_lines), 1):
            if "Direct configuration=" in all_lines[line]:
                x_num = x_num + 1
                if x_num > sstep:
                    lattice_i[inum, 0, 0:3] = all_lines[line - 5].split()[0:3]
                    lattice_i[inum, 1, 0:3] = all_lines[line - 4].split()[0:3]
                    lattice_i[inum, 2, 0:3] = all_lines[line - 3].split()[0:3]
                    stress_tensor[inum,:] = stress[x_num-1,:]
                    inum = inum + 1
        
        print("Start calculation of elastic constants and standard errors.")
        elas=np.zeros((n_num,6,6))
        V = np.zeros((n_num))
        Celas = np.zeros((6,6))
        elas_err = np.zeros((6,6))
        for k in np.arange(0,n_num):
            for i in np.arange(0, 3):
                for j in np.arange(0, 3):
                    lattice_avg[i, j] = np.mean(lattice_i[k*slice_step:(k+1)*slice_step, i, j])
            #print(lattice_avg)
            strain_tensor,volume_i = self.calc_strain(lattice_i[k*slice_step:(k+1)*slice_step, :, :], lattice_avg, index=slice_step)
            elas[k,:,:], V[k], title =self.calc_elas(strain=strain_tensor,stress=stress_tensor[k*slice_step:(k+1)*slice_step, :],volume=volume_i)
        for i in np.arange(0,6):
            for j in np.arange(0,6):
                Celas[i,j]=np.mean(elas[:,i,j])
                elas_err[i,j]=np.std(elas[:,i,j])
        
        print("Writing elastic constants and standard errors.")       
        #write standord error
        err_file = open("elastic_error_{}K.out".format(self.Temp), mode='w')
        for i in np.arange(0, 6, 1):
            print(
                format(elas_err[i, 0], ">10.4f"),
                format(elas_err[i, 1], ">10.4f"),
                format(elas_err[i, 2], ">10.4f"),
                format(elas_err[i, 3], ">10.4f"),
                format(elas_err[i, 4], ">10.4f"),
                format(elas_err[i, 5], ">10.4f"),
                file=err_file,
            )
        
        self.volume = np.mean(V[:])
        self.density = 10000 * float(self.mass) / (6.02 * self.volume)
        self.elasproperties(Celas=Celas, title=title)
        
    def read_nxdatcar(self, sstep=None, estep=None, nxdatcar=None):
        elas = np.zeros((nxdatcar,6,6))
        V= np.zeros((nxdatcar))
        print("Start reading the lattice matrix and stress tensor.")
        for n in np.arange(0,nxdatcar):
            lattice_avg = np.zeros((3, 3), dtype=np.float64)
            lattice_i = np.zeros((estep-sstep, 3, 3), dtype=np.float64)
            xdatacar = open("XDATCAR_{:02d}".format(n+1), mode="r")
            
            if os.path.isfile("stress_{:02d}.out".format(n+1)):
                pass
            else:
                if os.path.isfile("OUTCAR_{:02d}".format(n+1)):
                    shutil.copy("OUTCAR_{:02d}".format(n+1), "OUTCAR")
                    os.system("grep 'Total+kin' OUTCAR |awk '{print $2,$3,$4,$5,$6,$7}' >stress.out")
                    os.rename("stress.out", "stress_{:02d}.out".format(n+1))
                else:
                    print("Warning: No stress_*.out or OUTCAR_*")           
            stress_data = np.loadtxt("stress_{:02d}.out".format(n+1))
            print("Stress_{:02d} total step: {}; Start calculate step: {}".format(n+1,len(stress_data[:,0]), sstep))
            
            if estep > len(stress_data[:,0]):
                print("Input end_step > Total step")
                estep = len(stress_data[:,0])
            stress = np.zeros((estep-sstep, 6), dtype=np.float64)
            for i in np.arange(0,6):
                stress[:,i]=stress_data[sstep:estep,i]-np.mean(stress_data[sstep:estep,i])
            stress_tensor = np.zeros((estep-sstep, 6), dtype=np.float64)         
            stress_tensor = stress
            all_lines = xdatacar.readlines()
            inum = 0
            tstep = 0
            for line in np.arange(0, len(all_lines), 1):
                if "Direct configuration=" in all_lines[line]:
                    tstep = tstep + 1
                    if tstep>sstep and tstep <=estep:
                        lattice_i[inum, 0, 0:3] = all_lines[line - 5].split()[0:3]
                        lattice_i[inum, 1, 0:3] = all_lines[line - 4].split()[0:3]
                        lattice_i[inum, 2, 0:3] = all_lines[line - 3].split()[0:3]
                        inum = inum + 1
            
            lattice  = np.zeros((inum,3,3))
            print("Xdatcar_{:02d} total step: {}; Start calculate step: {}".format(n+1,tstep, sstep))
            if tstep != len(stress_data[:,0]):
                print("Error! stress numbers != strain numbers")
            
            if inum+sstep<estep:
                lattice =  lattice_i[0:inum,:,:]
            else:
                lattice = lattice_i
            for i in np.arange(0,3):
                for j in np.arange(0,3):
                    lattice_avg[i, j] = np.mean(lattice[:, i, j])
            
            strain_tensor, volume_i = self.calc_strain(lattice, lattice_avg, index=inum)
            elas[n,:,:], V[n], title =self.calc_elas(strain=strain_tensor,stress=stress_tensor,volume=volume_i)
        
        print("Start calculation of elastic constants and standard errors.")
        Celas = np.zeros((6,6))
        elas_err = np.zeros((6,6))
        for i in np.arange(0,6):
            for j in np.arange(0,6):
                Celas[i,j]=np.mean(elas[:,i,j])
                elas_err[i,j]=np.std(elas[:,i,j])
        
        #write standord error
        print("Writing elastic constants and standard errors.")
        err_file = open("elastic_error_{}K.out".format(self.Temp), mode='w')
        for i in np.arange(0, 6, 1):
            print(
                format(elas_err[i, 0], ">10.4f"),
                format(elas_err[i, 1], ">10.4f"),
                format(elas_err[i, 2], ">10.4f"),
                format(elas_err[i, 3], ">10.4f"),
                format(elas_err[i, 4], ">10.4f"),
                format(elas_err[i, 5], ">10.4f"),
                file=err_file,
            )
        
        self.volume = np.mean(V[:])
        self.density = 10000 * float(self.mass) / (6.02 * self.volume)
        self.elasproperties(Celas=Celas, title=title)
        
    def calc_strain(self, lattice_i, lattice_avg, index=None):
        lattice_avg_inv = np.linalg.inv(lattice_avg)
        volume_i = []
        strain_tensor = np.zeros((index, 6), dtype=np.float64)
        strain_matrix = np.zeros((index, 3, 3), dtype=np.float64)
        recipvect = np.array([[0.0, 0.0, 0.0]])
        for i in np.arange(0, index):
            strain_matrix[i, :, :] = np.matmul(lattice_avg_inv, lattice_i[i, :, :])
            strain_tensor[i, 0] = strain_matrix[i, 0, 0] - 1.0  # xx
            strain_tensor[i, 1] = strain_matrix[i, 1, 1] - 1.0  # yy
            strain_tensor[i, 2] = strain_matrix[i, 2, 2] - 1.0  # zz
            strain_tensor[i, 3] = strain_matrix[i, 1, 2]  # xy
            strain_tensor[i, 4] = strain_matrix[i, 0, 2]  # xz
            strain_tensor[i, 5] = strain_matrix[i, 0, 1]  # yz
            recipvect[0, 0] = (
                lattice_i[i, 1, 1] * lattice_i[i, 2, 2]
                - lattice_i[i, 2, 1] * lattice_i[i, 1, 2]
            )
            recipvect[0, 1] = (
                lattice_i[i, 1, 2] * lattice_i[i, 2, 0]
                - lattice_i[i, 2, 2] * lattice_i[i, 1, 0]
            )
            recipvect[0, 2] = (
                lattice_i[i, 1, 0] * lattice_i[i, 2, 1]
                - lattice_i[i, 2, 0] * lattice_i[i, 1, 1]
            )
            V = (
                lattice_i[i, 0, 0] * recipvect[0, 0]
                + lattice_i[i, 0, 1] * recipvect[0, 1]
                + lattice_i[i, 0, 2] * recipvect[0, 2]
            )
            volume_i.append(V)
        
        #self.calc_elas(strain=strain_tensor,volume=volume_i)
        return strain_tensor, volume_i
    
    def calc_elas(self, strain=None, stress=None, volume=None):
        Vol_avg = np.mean(volume)
        #print(len(stress[:,0]))
        # reload stress
        index = len(stress[:,0])
        S_stress = np.zeros((6*index,1))
        for i in np.arange(0,index):
            S_stress[6*i+0,0]=stress[i,0]
            S_stress[6*i+1,0]=stress[i,1]
            S_stress[6*i+2,0]=stress[i,2]
            S_stress[6*i+3,0]=stress[i,4]
            S_stress[6*i+4,0]=stress[i,5]
            S_stress[6*i+5,0]=stress[i,3]
            i = i+1
        
        # Reload strain matrix
        if self.spg_num >= 1 and self.spg_num <= 2:
            S_strain = np.zeros((6*index,21))
            for i in np.arange(0,index):
                S_strain[6*i+0,0:6]=strain[i,0:6]
                S_strain[6*i+1,1] = strain[i,0]; S_strain[6*i+1,6:11] = strain[i,1:5]
                S_strain[6*i+2,2] = strain[i,0]; S_strain[6*i+2,7]=strain[i,1]
                S_strain[6*i+2,11:15]=strain[i,2:6]
                S_strain[6*i+3,3] = strain[i,0]; S_strain[6*i+3,8]=strain[i,1]
                S_strain[6*i+3,12]=strain[i,2]; S_strain[6*i+3,15:18]=strain[i,3:6]
                S_strain[6*i+4,4]=strain[i,0]; S_strain[6*i+4,9]=strain[i,1]
                S_strain[6*i+4,13]=strain[i,2]; S_strain[6*i+4,16]=strain[i,3]
                S_strain[6*i+4,18:20] = strain[i,4:6]
                S_strain[6*i+5,5] = strain[i,0]; S_strain[6*i+5,10] = strain[i,1]
                S_strain[6*i+5,14] = strain[i,2]; S_strain[6*i+5,17] = strain[i,3]
                S_strain[6*i+5,19:21]=strain[i,4:6]
            solve_elas = np.linalg.lstsq(S_strain, S_stress,rcond=-1)[0]
            Elas, title = self.__cubic_solve(celas=-0.1*solve_elas)
#
        elif self.spg_num >= 3 and self.spg_num <= 15:
            S_strain = np.zeros((6*index,13))
            for i in np.arange(0,index):
                S_strain[6*i+0,0] = strain[i,0]; S_strain[6*i+0,1] = strain[i,1]
                S_strain[6*i+0,2] = strain[i,2]; S_strain[6*i+0,3] = strain[i,4]
                S_strain[6*i+1,1] = strain[i,0]; S_strain[6*i+1,4] = strain[i,1]
                S_strain[6*i+1,5] = strain[i,2]; S_strain[6*i+1,6] = strain[i,4]
                S_strain[6*i+2,2] = strain[i,0]; S_strain[6*i+2,5] = strain[i,1]
                S_strain[6*i+2,7] = strain[i,2]; S_strain[6*i+2,8] = strain[i,4]
                S_strain[6*i+3,9] = strain[i,3]; S_strain[6*i+3,10]= strain[i,5]
                S_strain[6*i+4,3] = strain[i,0]; S_strain[6*i+4,6] = strain[i,1]
                S_strain[6*i+4,8] = strain[i,2]; S_strain[6*i+4,11]= strain[i,4]
                S_strain[6*i+5,10]= strain[i,3]; S_strain[6*i+5,12]= strain[i,5]
            solve_elas = np.linalg.lstsq(S_strain, S_stress,rcond=-1)[0]
            Elas, title = self.__cubic_solve(celas=-0.1*solve_elas)
            
#
        elif self.spg_num >= 16 and self.spg_num <= 74:
            S_strain = np.zeros((6*index,9))
            for i in np.arange(0,index):
                S_strain[6*i+0,0:3] = strain[i,0:3]
                S_strain[6*i+1,1] = strain[i,0]; S_strain[6*i+1,3:5] = strain[i,1:3]
                S_strain[6*i+2,2] = strain[i,0]; S_strain[6*i+2,4:6] = strain[i,1:3]
                S_strain[6*i+3,6] = strain[i,3]; S_strain[6*i+4,7] = strain[i,4]
                S_strain[6*i+5,8] = strain[i,5]
            
            solve_elas = np.linalg.lstsq(S_strain, S_stress,rcond=-1)[0]
            Elas, title = self.__orthorhombic_solve(celas=-0.1*solve_elas)

#
        elif self.spg_num >= 75 and self.spg_num <= 88:
            #11 12 13 16 33 44 66
            S_strain = np.zeros((6*index,7))
            for i in np.arange(0,index):
                S_strain[6*i+0,0:3] = strain[i,0:3]; S_strain[6*i+0,3] = strain[i,5]
                S_strain[6*i+1,1] = strain[i,0]; S_strain[6*i+1,0] = strain[i,1]
                S_strain[6*i+1,2] = strain[i,2]; S_strain[6*i+1,3] = -strain[i,5]
                S_strain[6*i+2,2] = strain[i,0]+strain[i,1]; S_strain[6*i+2,4] = strain[i,2]
                S_strain[6*i+3,5] = strain[i,3]; S_strain[6*i+4,5] = strain[i,4]
                S_strain[6*i+5,3] = strain[i,0]-strain[i,1]; S_strain[6*i+5,7] = strain[i,5]
            solve_elas = np.linalg.lstsq(S_strain, S_stress,rcond=-1)[0]
            Elas, title = self.__cubic_solve(celas=-0.1*solve_elas)
#
        elif self.spg_num >= 89 and self.spg_num <= 142:
            #111213334466
            S_strain = np.zeros((6*index,6))
            for i in np.arange(0,index):
                S_strain[6*i+0,0:3] = strain[i,0:3]
                S_strain[6*i+1,0] = strain[i,1]; S_strain[6*i+1,1] = strain[i,0]
                S_strain[6*i+1,2] = strain[i,2]; S_strain[6*i+2,2] = strain[i,0]+strain[i,1]
                S_strain[6*i+2,3] = strain[i,2]
                S_strain[6*i+3,4] = strain[i,3]; S_strain[6*i+4,4] = strain[i,4]
                S_strain[6*i+5,5] = strain[i,5]
            solve_elas = np.linalg.lstsq(S_strain, S_stress,rcond=-1)[0]
            Elas, title = self.__cubic_solve(celas=-0.1*solve_elas)
                
#
        elif self.spg_num >= 143 and self.spg_num <= 148:
            #11 12 13 14 15 33 44 45
            S_strain = np.zeros((6*index,8))
            for i in np.arange(0,index):
                S_strain[6*i+0,0:5] = strain[i,0:5]
                S_strain[6*i+1,0] = strain[i,1]; S_strain[6*i+1,1] = strain[i,0]
                S_strain[6*i+1,2] = strain[i,2]; S_strain[6*i+1,3] = -strain[i,3]
                S_strain[6*i+1,4] = -strain[i,4]
                S_strain[6*i+2,2] = strain[i,0]+strain[i,1]
                S_strain[6*i+2,5] = strain[i,2]
                S_strain[6*i+3,3] = strain[i,0]-strain[i,1]
                S_strain[6*i+3,6] = strain[i,3]; S_strain[6*i+3,7] = -strain[i,5]
                S_strain[6*i+4,3] = strain[i,5]; S_strain[6*i+4,6] = strain[i,4]
                S_strain[6*i+4,4] = strain[i,0] - strain[i,1]
                S_strain[6*i+5,0] = 0.5*strain[i,5]; S_strain[6*i+5,1] = -0.5*strain[i,5]
                S_strain[6*i+5,3] = strain[i,4]; S_strain[6*i+5,7] = -strain[i,3]
            solve_elas = np.linalg.lstsq(S_strain, S_stress,rcond=-1)[0]
            Elas, title = self.__cubic_solve(celas=-0.1*solve_elas)
#       
        elif self.spg_num >= 148 and self.spg_num <= 167:
            #11 12 13 14 33 44
            S_strain = np.zeros((6*index,6))
            for i in np.arange(0,index):
                S_strain[6*i+0,0:4] = strain[i,0:4]
                S_strain[6*i+1,0] = strain[i,1]; S_strain[6*i+1,1] = strain[i,0]
                S_strain[6*i+1,2] = strain[i,2]; S_strain[6*i+1,3] = -strain[i,3]
                S_strain[6*i+2,2] = strain[i,0]+strain[i,1]
                S_strain[6*i+2,4] = strain[i,2]
                S_strain[6*i+3,3] = strain[i,0]-strain[i,1]
                S_strain[6*i+3,5] = strain[i,3]
                S_strain[6*i+4,3] = strain[i,5]
                S_strain[6*i+4,5] = strain[i,4]
                S_strain[6*i+5,0] = 0.5*strain[i,5]; S_strain[6*i+5,1] = -0.5*strain[i,5]
                S_strain[6*i+5,3] = strain[i,4]
            solve_elas = np.linalg.lstsq(S_strain, S_stress,rcond=-1)[0]
            Elas, title = self.__cubic_solve(celas=-0.1*solve_elas)
                
                
        elif self.spg_num >= 168 and self.spg_num <= 194:
            S_strain = np.zeros((6*index,5))
            for i in np.arange(0,index):
                S_strain[6*i+0,0:3] = strain[i,0:3]
                S_strain[6*i+1,0] = strain[i,1]; S_strain[6*i+1,1] = strain[i,0]
                S_strain[6*i+1,2] = strain[i,2]
                S_strain[6*i+2,2] = strain[i,0]+strain[i,1]
                S_strain[6*i+2,3] = strain[i,2]
                S_strain[6*i+3,4] = strain[i,3]
                S_strain[6*i+4,4] = strain[i,4]
                S_strain[6*i+5,0] = 0.5*strain[i,5]
                S_strain[6*i+5,1] = -0.5*strain[i,5]
            solve_elas = np.linalg.lstsq(S_strain, S_stress,rcond=-1)[0]
            Elas, title = self.__cubic_solve(celas=-0.1*solve_elas)

        elif self.spg_num >= 195 and self.spg_num <= 230:
            S_strain = np.zeros((6*index,3))
            for i in np.arange(0,index):
                S_strain[6*i+0,0]=strain[i,0]
                S_strain[6*i+0,1]=strain[i,1]+strain[i,2]
                S_strain[6*i+1,0]=strain[i,1]
                S_strain[6*i+1,1]=strain[i,0]+strain[i,2]
                S_strain[6*i+2,0]=strain[i,2]
                S_strain[6*i+2,1]=strain[i,0]+strain[i,1]
                S_strain[6*i+3,2]=strain[i,3]
                S_strain[6*i+4,2]=strain[i,4]
                S_strain[6*i+5,2]=strain[i,5]
            
            solve_elas = np.linalg.lstsq(S_strain, S_stress,rcond=-1)[0]
            Elas, title = self.__cubic_solve(celas=-0.1*solve_elas)
            #print(-0.1*Elas[1])
        
        return Elas, Vol_avg, title 
    
    def __cubic_solve(self, celas=None):
        
        elas = np.zeros((6, 6))
        
        elas[0, 0] = celas[0]
        elas[0, 1] = celas[1]
        elas[3, 3] = celas[2]
        
        elas[4, 4] = elas[5, 5] = elas[3, 3]
        elas[1, 0] = elas[0, 2] = elas[2, 0] = elas[1, 2] = elas[2, 1] = elas[0, 1]
        elas[1, 1] = elas[2, 2] = elas[0, 0]

        Title = "The cubic crystal mechanical properties"

        return elas, Title
        
    def __hexagonal_solve(self, celas=None):
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
        
        elas[0, 0] = celas[0]
        elas[0, 1] = celas[1]
        elas[0, 2] = celas[2]
        
        elas[2, 2] = celas[3]
        elas[3, 3] = celas[4]
        
        elas[1, 1] = elas[0, 0]
        elas[1, 0] = elas[0, 1]
        elas[2, 0] = elas[1, 2] = elas[2, 1] = elas[0, 2]
        elas[4, 4] = elas[3, 3]
        elas[5, 5] = (elas[0, 0] - elas[0, 1]) / 2.0

        Title = "The hexagonal crystal mechanical properties"

        return elas, Title
    
    def __rhombohedral_I_solve(self, celas=None):
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
        
        elas[0, 0] = celas[0]
        elas[0, 1] = celas[1]
        elas[0, 2] = celas[2]
        elas[0, 3] = celas[3]
        
        elas[2, 2] = celas[4]
        elas[3, 3] = celas[5]

        elas[1, 1] = elas[0, 0]
        elas[1, 0] = elas[0, 1]
        elas[2, 0] = elas[1, 2] = elas[2, 1] = elas[0, 2]
        elas[4, 4] = elas[3, 3]
        elas[5, 5] = (elas[0, 0] - elas[0, 1]) / 2.0
        elas[3, 0] = elas[4, 5] = elas[5, 4] = elas[0, 3]
        elas[1, 3] = elas[3, 1] = -elas[0, 3]

        Title = "The rhombohedral I crystal mechanical properties"

        return elas, Title

    def __rhombohedral_II_solve(self, celas=None):
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
        
        elas[0, 0] = celas[0]
        elas[0, 1] = celas[1]
        elas[0, 2] = celas[2]
        elas[0, 3] = celas[3]
        elas[0, 4] = celas[4]
        
        elas[2, 2] = celas[5]
        elas[3, 3] = celas[6]

        elas[1, 1] = elas[0, 0]
        elas[1, 0] = elas[0, 1]
        elas[2, 0] = elas[1, 2] = elas[2, 1] = elas[0, 2]
        elas[4, 4] = elas[3, 3]
        elas[5, 5] = (elas[0, 0] - elas[0, 1]) / 2.0
        elas[3, 0] = elas[4, 5] = elas[5, 4] = elas[0, 3]
        elas[1, 3] = elas[3, 1] = -elas[0, 3]
        elas[1, 4] = elas[4, 1] = -elas[0, 4]
        elas[4, 0] = elas[0, 4]
        elas[5, 3] = elas[3, 5] = celas[7]
        
        Title = "The rhombohedral II crystal mechanical properties"

        return elas, Title

    def __tetragonal_I_solve(self, celas=None):
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
        
        elas[0, 0] = celas[0]
        elas[0, 1] = celas[1]
        elas[0, 2] = celas[2]
        
        elas[2, 2] = celas[3]
        elas[3, 3] = celas[4]
        
        elas[1, 1] = elas[0, 0]
        elas[1, 0] = elas[0, 1]
        elas[2, 0] = elas[1, 2] = elas[2, 1] = elas[0, 2]
        elas[4, 4] = elas[3, 3]
        elas[5, 5] = celas[5]
        
        Title = "The tetragonal I crystal mechanical properties"

        return elas, Title

    def __tetragonal_II_solve(self, celas=None):
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
        
        elas[0, 0] = celas[0]
        elas[0, 1] = celas[1]
        elas[0, 2] = celas[2]
        elas[0, 5] = celas[3]
        
        elas[2, 2] = celas[4]
        elas[3, 3] = celas[5]
        
        elas[1, 1] = elas[0, 0]
        elas[1, 0] = elas[0, 1]
        elas[2, 0] = elas[1, 2] = elas[2, 1] = elas[0, 2]
        elas[4, 4] = elas[3, 3]
        elas[5, 5] = celas[6]
        elas[1, 5] = -elas[0, 5]
        
        Title = "The tetragonal II crystal mechanical properties"

        return elas, Title

    def __orthorhombic_solve(self, celas=None):
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
        
        elas[0, 0] = celas[0]
        elas[0, 1] = celas[1]
        elas[0, 2] = celas[2]
        
        elas[1, 1] = celas[3]
        elas[1, 2] = celas[4]
        
        elas[2, 2] = celas[5]
        elas[3, 3] = celas[6]
        elas[4, 4] = celas[7]
        elas[5, 5] = celas[8]
        
        elas[1, 0] = elas[0, 1]
        elas[2, 1] = elas[1, 2]
        elas[2, 0] = elas[0, 2]

        Title = "The orthorhombic crystal mechanical properties"

        return elas, Title

    def __monoclinic_solve(self, celas=None):

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
        
        elas[0, 0] = celas[0]
        elas[0, 1] = celas[1]
        elas[0, 2] = celas[2]
        elas[0, 4] = celas[3]
        
        elas[1, 1] = celas[4]
        elas[1, 2] = celas[5]
        elas[1, 4] = celas[6]
        
        elas[2, 2] = celas[7]
        elas[2, 4] = celas[8]
        elas[3, 3] = celas[9]
        elas[3, 5] = celas[10]
        
        elas[4, 4] = celas[11]
        elas[5, 5] = celas[12]
        
        elas[1, 0] = elas[0, 1]
        elas[2, 1] = elas[1, 2]
        elas[2, 0] = elas[0, 2]
        elas[4, 0] = elas[0, 4]
        elas[4, 1] = elas[1, 4]
        elas[4, 2] = elas[2, 4]
        elas[5, 3] = elas[3, 5]
        
        Title = "The monoclinic crystal mechanical properties"

        return elas, Title

    def __triclinic_solve(self, celas=None):
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
        
        elas[0, 0] = celas[0]
        elas[0, 1] = celas[1]
        elas[0, 2] = celas[2]
        elas[0, 3] = celas[3]
        elas[0, 4] = celas[4]
        elas[0, 5] = celas[5]
        
        elas[1, 1] = celas[6]
        elas[1, 2] = celas[7]
        elas[1, 3] = celas[8]
        elas[1, 4] = celas[9]
        elas[1, 5] = celas[10]
        
        elas[2, 2] = celas[11]
        elas[2, 3] = celas[12]
        elas[2, 4] = celas[13]
        elas[2, 5] = celas[14]
        
        elas[3, 3] = celas[15]
        elas[3, 4] = celas[16]
        elas[3, 5] = celas[17]
        
        elas[4, 4] = celas[18]
        elas[4, 5] = celas[19]
        
        elas[5, 5] = celas[20]
        
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

        return elas, Title
    
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
    
    def elasproperties(self, Celas=None, title=None):

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
        elasfile = open("second_elastic_{}K.out".format(self.Temp), mode="w")

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
                format(Celas[i, 0], ">10.3f"),               
                format(Celas[i, 1], ">10.3f"),               
                format(Celas[i, 2], ">10.3f"),               
                format(Celas[i, 3], ">10.3f"),                
                format(Celas[i, 4], ">10.3f"),                
                format(Celas[i, 5], ">10.3f"),
                file=elasfile,
            )
        print("\n", end="", file=elasfile)
        print("Compliance tensor S_ij (unit: GPa^-1)", file=elasfile)

        for i in np.arange(0, 6, 1):
            print(
                format(Selas[i, 0], ">10.6f"),
                format(Selas[i, 1], ">10.6f"),
                format(Selas[i, 2], ">10.6f"),
                format(Selas[i, 3], ">10.6f"),
                format(Selas[i, 4], ">10.6f"),
                format(Selas[i, 5], ">10.6f"),
                file=elasfile,
            )

        print("\n", end="", file=elasfile)
        print("mechanical stability:  " + if_stable, file=elasfile)

        print("\n", end="", file=elasfile)

        print("supercell volume :  {:.4f} A^3".format(self.volume), file=elasfile)
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
        
        print("\n", end="", file=elasfile)
        print("Please cite: Comput. Phys. Commun., 281 (2022), 108495", file=elasfile)
        print("             Comput. Phys. Commun., 273 (2022), 108280.", file=elasfile)

if __name__ == "__main__":
    #npt_solve_method_1(Temp=300).read_sxdatcar(sstep=10000, slice_step=60000)
    npt_solve_method_1(Temp=300).read_nxdatcar(sstep=10000, estep=100000, nxdatcar=1)


                
        
        
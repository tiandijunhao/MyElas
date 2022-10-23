#!/bin/bash

# Relax structure
mkdir Relax
cd Relax/
cp ../POTCAR ./
cp ../POSCAR ./
Myelas --vasp incar --encut 600.0 --isif 3 --kspace 0.10 -ctype rlx
mpirun -np 32 vasp_std >vasp_relax.log
cp CONTCAR ../POSCAR-uc
cd ..

# Elastic calculation
Myelas -g 3D_2nd -smax 0.018 -snum 13
root_path=$(pwd)
for i in nelastic_*; do
    cd ${i}
    for j in strain_*; do

        cd ${j}
        cp ${root_path}/POTCAR ./

        # IONIC position relax
        Myelas --vasp incar --encut 600.0 --isif 2 --kspace 0.10 -ctype rlx
        time mpirun -np 32 vasp_std >vasp_relax.log

        cp CONTCAR ../POSCAR

        # Static calculation
        Myelas --vasp incar --encut 600.0 --isif 2 --kspace 0.10 -ctype stc

        # Static calculation for electron temperature
        #Myelas --vasp incar --encut 600.0 --isif 2 --kspace 0.10 -Te 300.0 -ctype stc
        time mpirun -np 32 vasp_std >vasp_stc.log

        cd ..
    done
    cd ..
done

# Post-processing
Myelas -so 3D_2nd -smax 0.018 -snum 13

# Visualization
Myelas -p3D Youngs -ptype plane              #plot Young's modulus
Myelas -p3D Shear -ptype plane -minmax max   #plot Max Shear modulus
Myelas -p3D Shear -ptype plane -minmax min   #plot Max Shear modulus
Myelas -p3D Poisson -ptype plane -minmax max #plot Max Shear modulus
Myelas -p3D Poisson -ptype plane -minmax min #plot Max Shear modulus
Myelas -p3D SV                               #plot single-crystalline sound velocity

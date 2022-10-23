#!/bin/bash
rm WAVECAR
for i in $(seq 450 50 800)
do
cat > INCAR <<!
SYSTEM = ThTe
PREC = Accurate
ISTART = 0
ICHARG = 2
EDIFF = 1E-8 
EDIFFG = -0.001
ENCUT = $i
ISMEAR = -5
SIGMA  = 0.1
#IBRION = 2
ISIF = 2
#NSW = 100
#POTIM = 0.5
NELM = 100
GGA=PE 
#ISPIN = 1
ISYM = 2
#ALGO = Normal
LREAL= .FALSE.
LCHARG = .FALSE.
LWAVE = .FALSE.
KSPACING = 0.10 
!
echo "ENCUT = $i eV" ; time mpirun -n 16 vasp_std
E=$(grep "TOTEN" OUTCAR | tail -1 | awk '{printf "%12.9f \n", $5 }')
echo $i $E >>ecut_test.out
done

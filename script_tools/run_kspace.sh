#!/bin/bash
rm WAVECAR
for i in  $(seq 0.5 -0.05 0.05) 
do
cat > INCAR <<!
SYSTEM= CeO2
ENCUT = 650
GGA=PE
ADDGRID= .TRUE.
ISTART=0
ICHARG = 2
EDIFF= 1E-8
EDIFFG= -0.001
ISMEAR = -5
PREC = Accurate
KSPACING = $i
#KGAMMA =
!
echo "k = $i eV" ; time mpirun -n 16 vasp_std
E=`tail -1 OSZICAR`
echo $i  $E >>kspace_test.out
done

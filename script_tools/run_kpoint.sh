#!/bin/sh
rm WAVECAR
for i in $(seq 6 1 18)
  do
  cat > INCAR <<!
SYSTEM= ThSe
ENCUT = 650
GGA=PE
ADDGRID= .TRUE.
ISTART=0
ICHARG = 2
EDIFF= 1E-8
EDIFFG= -0.001
ISMEAR = -5
PREC = Accurate
!
cat > KPOINTS <<!
Accurate
0
G
$i $i $i
!
echo "k mesh = $i x $i x $i" ; time mpirun -n 16 vasp_std
E=`tail -1  OSZICAR`
echo $i  $E >>kpoint_test.out
done
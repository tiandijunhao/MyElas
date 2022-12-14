# **Myelas user manual**

[TOC]

<div STYLE="page-break-after: always;"></div>

## **1 How to install Myelas**

```bash
tar -zxvf MyElas.tar.gz
cd MyElas
python setup.py install
```





## **2  Module and Script**

### **2.1 Module introduction**

**VASP input module (para_vasp.py):** Generate KPOINT files and INCAR files .

**Read POSCAR module (read_poscar.py):** Read the POSCAR-uc file.

**Read elastic tensor module (read_elastic.py):** Read the elastic tensor from the OUTCAR or elastic.out.

**Generate 2D strain POSCAR module (strain_poscar_2D.py):** Generate 2D POSCAR of different strains required for calculation.

**Generate 3D strain POSCAR module (strain_poscar_3D.py):** Generate 3D POSCAR of different strains required for calculation.

**Generate 3D third strain POSCAR module (strain_poscar_3rd3D.py):** Generate 3D POSCAR of different strains required for the third elastic constants calculation.

**Solve the second elastic of 2D module (solve_elastic_2D.py):** Used to calculate the second elastic constants and related physical quantities of 2D materials.

**Solve the second elastic of 3D module (solve_elastic_3D.py):** Used to calculate the second elastic constants and related physical quantities  of 3D materials.

**Solve the third elastic of 3D module (solve_elastic_3rd3D.py):** Used to calculate the third elastic constants and related physical quantities  of 3D materials.

**Plot the fitting E-V figure (plot_fit_E.py):** Plotting the fitting EV figure of each strain matrix.

**Plot the 3D figure (plot_projiect.py plot_project_cubic.py):** Visualization of Young’s, Bulk, Shear,  Poisson ratio, and sound velocity of single-crystal materials.

**Calculate elastic constants from phonon spectral data (phonon_to_elas.py):**  Support the phonon spectral data which generate by TDEP, Alamode and phonopy software.



### **2.2 Script tool**

Some scripts are in the *script_tools* folder.

**auto_elas_calculation.sh**: Automatic calculation process of elastic constant

```shell
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
```



<div STYLE="page-break-after: always;"></div>

## **3 Introduction to input and output files**

### **3.1 Input file**

**POSCAR-uc:** The optimized initial stable structure. The primitive cell or the conventional unit cell. Same format as POSCAR.

example:

```txt
strain_poscar                           
   1.00000000000000     
     2.3457218738000001    0.0000000000000000    0.0000000000000000
    -1.1728609369000000    2.0314547329999999    0.0000000000000000
     0.0000000000000000    0.0000000000000000    3.7183867462000002
   Fe
     2
Direct
  0.6666666667000030  0.3333333332999970  0.7500000000000000
  0.3333333332999970  0.6666666667000030  0.2500000000000000
```



**POTCAR:**  The potential file of VASP.



**Phonon.dat:** The phonon dispersion data. If we use the “Myelas -rp” command, this file will be needed.



**input_direct** : If we use the “Myelas -rp” command, this file will be needed.

example:  Please set the value according to the actual high symmetry path.

```txt
# Direction  Position left(-1)/right(1)
Cubic:
[100] ****** -1
[110] ****** 1
[111] ****** 1

Hexagonal and Trigonal:
[100] ***** 1
[001] ***** -1

Tetragonal：
[100] ******* -1
[001] ******* 1
[110] ******* 1

Orthorhombic：
[100] ******* 1
[010] ******* -1
[001] ******* 1

```



**OUTCAR** or **elastic.out**  : The file which include elsatic tensor. The OUTCAR is generated by VASP.

example (elastic.out ):

```
296.036    61.286    26.809    0.000    0.000    0.000
61.286    220.654    146.618    0.000    0.000    0.000
26.809    146.618    349.260    0.000    0.000    0.000
0.000    0.000    0.000    152.599    0.000    0.000
0.000    0.000    0.000    0.000    124.848    0.000
0.000    0.000    0.000    0.000    0.000    102.399
```



### **3.2 Output file**

**RE-POSCAR:**  The normalized unit cell. Used to generate the strained POSCAR file.

example:

```txt
recell_poscar
1.0
2.3457218738     0.0000000000     0.0000000000
-1.1728609369     2.0314547330     0.0000000000
0.0000000000     0.0000000000     3.7183867462
Fe 
2 
Direct
0.6666666667     0.3333333333     0.7500000000
0.3333333333     0.6666666667     0.250000000
```





**E_Strain.out:** The data used for polynomial fitting and the data regenerated by the parameters obtained by polynomial fitting.

example:

```
 Space group 227
#nelastic 01  strain  energy  (E-E0)/V0
-0.018000  -43.36257630  0.00022698
-0.015000  -43.37391881  0.00015762
-0.012000  -43.38320566  0.00010083
-0.009000  -43.39042023  0.00005671
-0.006000  -43.39557190  0.00002521
-0.003000  -43.39866221  0.00000631
-0.000000  -43.39969451  0.00000000
0.003000  -43.39866361  0.00000630
0.006000  -43.39557542  0.00002519
0.009000  -43.39043256  0.00005664
0.012000  -43.38323442  0.00010066
0.015000  -43.37398325  0.00015723
0.018000  -43.36268100  0.00022634

#nelastic 02  strain  energy  (E-E0)/V0
-0.018000  -43.32686550  0.00044536
-0.015000  -43.34936909  0.00030774
-0.012000  -43.36762455  0.00019611
-0.009000  -43.38170239  0.00011002
-0.006000  -43.39168722  0.00004897
-0.003000  -43.39765857  0.00001245
-0.000000  -43.39969451  0.00000000
0.003000  -43.39786887  0.00001116
0.006000  -43.39226253  0.00004545
0.009000  -43.38295199  0.00010238
0.012000  -43.37000607  0.00018155
0.015000  -43.35349941  0.00028249
0.018000  -43.33351492  0.00040469

#nelastic 03  strain  energy  (E-E0)/V0
-0.018000  -43.25833744  0.00086441
-0.015000  -43.30238628  0.00059505
-0.012000  -43.33792058  0.00037775
-0.009000  -43.36518241  0.00021104
-0.006000  -43.38441393  0.00009344
-0.003000  -43.39583818  0.00002358
-0.000000  -43.39969451  0.00000000
0.003000  -43.39619123  0.00002142
0.006000  -43.38555478  0.00008647
0.009000  -43.36799921  0.00019382
0.012000  -43.34373092  0.00034222
0.015000  -43.31294881  0.00053046
0.018000  -43.27586102  0.00075725

```





**second_elastic.out:** Elastic constants and related properties

example (alpha-Uranium):

```shell
The orthorhombic crystal mechanical properties
Orthorhombic crystal (spacegroup No.: 63)

Elastic tensor C_ij (unit: GPa)
   296.036    61.286    26.809    0.000    0.000    0.000
   61.286    220.654    146.618    0.000    0.000    0.000
   26.809    146.618    349.260    0.000    0.000    0.000
   0.000    0.000    0.000    152.599    0.000    0.000
   0.000    0.000    0.000    0.000    124.848    0.000
   0.000    0.000    0.000    0.000    0.000    102.399

Compliance tensor S_ij (unit: GPa^-1)
   0.003594    -0.001130    0.000199    0.000000    0.000000    0.000000
   -0.001130    0.006641    -0.002701    0.000000    0.000000    0.000000
   0.000199    -0.002701    0.003982    0.000000    0.000000    0.000000
   0.000000    0.000000    0.000000    0.006553    0.000000    0.000000
   0.000000    0.000000    0.000000    0.000000    0.008010    0.000000
   0.000000    0.000000    0.000000    0.000000    0.000000    0.009766

mechanical stability:  Stable

unit cell volume :  80.3273 A^3
unit cell density:  19689.2973 kg/m^3

Polycrystalline modulus
(Unit: GPa) Bulk modulus    Shear modulus    Youngs modulus    Possion ratio    P-wave modulus
  Vogit      148.3750           118.0517           279.9180           0.1856           305.7772
  Reuss      143.8584           103.8917           251.2037           0.2090           282.3807
  Hill       146.1167           110.9717           265.5608           0.1973           294.0790

Cauchy Pressure  (GPa):  -91.3127
Pugh's ratio          :  0.7595
Vickers hardness (GPa):  19.7866

Anisotropy index:
  Chung-Buessem anisotropy index:  0.06
  Universal anisotropy index    :  0.71
  Log-Euclidean anisotropy index:  0.29

Polycrystalline sound velocity (m/s)
  Longitudinal sound velocity:  3864.7097
  Shear sound velocity       :  2374.0564
  Bulk sound velocity        :  2724.1738
  Average sound velocity     :  2620.0689

Pure single-crystal sound velocity (m/s)
  [100] direction:  vl = 3877.550  vs1 = 2280.517  vs2 = 2518.116
  [010] direction:  vl = 3347.654  vs1 = 2280.517  vs2 = 2783.944
  [001] direction:  vl = 4211.716  vs1 = 2518.116  vs2 = 2783.944

Debye temperature:  287.02 K

The minimum thermal conductivity:
  Clark model  :  0.597 W/(m K)
  Chaill model :  0.649 W/(m K)

```





**ELADAT_3rd**:

```
The third elastic constants.
Please check the number of elastic constants for your structure.


C111 =  -746.5459
C112 =  -438.7388
C123 =  -92.4655


C144 =  59.9915
C166 =  -315.3537
C456 =  -71.2529

```



**Nelastic_*.png:** the fitting EV figure of each strain matrix.

![nelas](F:\王豪-论文\写作\myelas\Fig\nelas.png)



**phonon_elastic.out:** The elastic constants which calculated from long wavelength limit.

example (alpha-U ):

```txt
Orthorhombic crystal:

C11  C22  C33  C44  C55  C66 (GPa)
354.655   257.168   402.203   143.639   120.159   89.186

Single-crystalline sound velocity (m/s)
100  vl = 4242.888;  vs1 = 2088.201; vs2 = 2411.127
010  vl = 3612.992;  vs1 = 2166.440; vs2 = 2665.254
001  vl = 4518.362;  vs1 = 2526.833; vs2 = 2734.691

unit cell volume :  80.2805 A^3
unit cell density:  19700.7659 kg/m^3

```



**Elastic modulus and sound velocity of single crystal figure:**

example:

![](F:\王豪-论文\写作\myelas\MyElas\example\bulk-materials-2nd-elastic-constants\Si\Young's modulus.png)

![](F:\王豪-论文\写作\myelas\MyElas\example\bulk-materials-2nd-elastic-constants\Si\Max poisson ratio.png)

![Single Sound Velocity](F:\王豪-论文\写作\myelas\Single Sound Velocity.png)

<div STYLE="page-break-after: always;"></div>

## **4. Command option**

```shell
usage: Myelas [-h] [-g 3D_2nd/2D_2nd/3D_3rd] [-so 3D_2nd/2D_2nd/3D_3rd] [-smax 0.018] [-snum 13]
              [-p3D Youngs/Bulk/Shear/Poisson/SV] [-p2D 2D] [-ptype 3D/plane] [-minmax min/max]
              [--vasp incar/kpoints/2D_kpoints] [--encut 600.0] [-Te 300.0] [-p 1000.0] 
              [--ismear 0] [--sigma 0.05] [--isif 2]
              [-ks 0.10] [-kp 6] [-kt G/M] [-ctype rlc/stc] [-re OUTCAR/elastic.out] 
              [-rp phonopy/alamode/alamode_scph/tdep]
              [-pf alamode.bands]

myelas optionn

optional arguments:
  -h, --help            show this help message and exit
  -g 3D_2nd/2D_2nd/3D_3rd, --generate 3D_2nd/2D_2nd/3D_3rd
                        Generate elastic type which need calculate
  -so 3D_2nd/2D_2nd/3D_3rd, --solve 3D_2nd/2D_2nd/3D_3rd
                        Solve elastic type which need calculate
  -smax 0.018, --strainmax 0.018
                        The strain max number
  -snum 13, --strainnum 13
                        The strain number
  -p3D Youngs/Bulk/Shear/Poisson/SV, --plot_3D Youngs/Bulk/Shear/Poisson/SV
                        plot single crystal modulus
  -p2D 2D, --plot_2D 2D
                        plot 2D materials modulus
  -ptype 3D/plane, --plot_3D_type 3D/plane
                        plot type
  -minmax min/max, --minmax min/max
                        plot maxnum or minnum for shear and poisson ratio
  --vasp incar/kpoints/2D_kpoints
                        Generate the VASP input file
  --encut 600.0         The encut energy in INCAR
  -Te 300.0, --Temperature 300.0
                        The electron temperature (K) in INCAR
  -p 1000.0, --pressure 1000.0
                        The pressure in INCAR, unit: GPa, 1GPa=10kB
  --ismear 0            The ISMEAR in INCAR
  --sigma 0.05          The smearing width in VASP
  --isif 2              The ISIF in INCAR
  -ks 0.10, --kspace 0.10
                        The KSPACING value in VASP
  -kp 6, --kpoint 6     The kpoint number in KPINTS
  -kt G/M, --kpoint_type G/M
                        The kpoint type
  -ctype rlc/stc, --calc_type rlc/stc
                        The calculation type : relax (rlx) or static (stc)
  -re OUTCAR/elastic.out, --read_elastic OUTCAR/elastic.out
                        Read the elastic tensor from OUTCAR or elastic.out file
  -rp phonopy/phonopy_old/alamode/alamode_scph/tdep, --read_phonon_to_calc phonopy/alamode/alamode_scph/tdep
                        Read the phonon dispersiaon to calculate elastic constants
  -pf alamode.bands, --input_phonon_file alamode.bands
                        input phonon data file


```



Generate the INCAR or KPOINTS

```shell
# Relax cell
Myelas --vasp incar --encut 600.0 --isif 3 -ctype rlx

# Static calculation
Myelas --vasp incar --encut 600.0  -ctype stc

# Include electron temperature (300 K)
Myelas --vasp incar --encut 600.0 -Te 300.0 -ctype stc

# KPOINTS (Gamma mesh 12 12 12)
Myelas --vasp kpoints -kp 12 -kt G
```



Generate the strain POSCAR

```shell
Myelas -g 3D_2nd -smax 0.018 -snum 13 # The second-order elastic constants of 3D solid materials
Myelas -g 2D_2nd -smax 0.018 -snum 13 # The second-order elastic constants of 2D solid materials
Myelas -g 3D_3rd -smax 0.060 -snum 25 # The third-order  elastic constants of 3D solid materials
```



Solve the elastic constants

```shell
Myelas -so 3D_2nd -smax 0.018 -snum 13 # The second-order elastic constants of 3D solid materials
Myelas -so 2D_2nd -smax 0.018 -snum 13 # The second-order elastic constants of 2D solid materials
Myelas -so 3D_3rd -smax 0.060 -snum 25 # The third-order  elastic constants of 3D solid materials
```



Calculate the elastic constants from phonon spectral data

```
Myelas -rp phonopy -pf band.yaml
Myelas -rp alamode -pf alamode.bands
Myelas -rp alamode_scph -pf alamode.bands
Myelas -rp tdep -pf outfile.dispersion_relations
```



The elastic tensor is read directly from the OUTCAR or elastic.out file to calculate the relevant properties.

```shell
Myelas -re OUTCAR
Myelas -re elastic.out
```



Visualization

```shell
# The Youngs modulus in sphere space
Myelas -p3D Youngs -ptype 3D 

# The maxnum Shear modulus in sphere space
Myelas -p3D Shear -ptype 3D -minmax max

# Distribution of Young's modulus on plane 
Myelas -p3D Youngs -ptype plane

# The single-crystalline sound velocity in sphere space
Myelas -p3D SV
```



<div STYLE="page-break-after: always;"></div>

## **5. Example**

### **5.1 Si elastic constants**

#### **5.1.1 Elastic constants calculation**

**Input file:**

POSCAR-uc:

```
Si8                                     
1.00000000000000     
  5.4684663969085214    0.0000000000000000    0.0000000000000000
  0.0000000000000000    5.4684663969085214    0.0000000000000000
  0.0000000000000000    0.0000000000000000    5.4684663969085214
Si
8
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.2500000000000000  0.7500000000000000  0.7500000000000000
  0.5000000000000000  0.0000000000000000  0.5000000000000000
  0.0000000000000000  0.5000000000000000  0.5000000000000000
  0.5000000000000000  0.5000000000000000  0.0000000000000000
  0.7500000000000000  0.2500000000000000  0.7500000000000000
  0.7500000000000000  0.7500000000000000  0.2500000000000000
  0.2500000000000000  0.2500000000000000  0.2500000000000000

```





1. **Generate the strain POSCAR:**

```shell
Myelas -g 3D_2nd -smax 0.018 -snum 13 # Only the second elastic constants
Myelas -g 3D_3rd -smax 0.060 -snum 13 # The third elastic constants
```





2. **DFT  calculation:**

Third elastic constants calculation example:

```shell
#!/bin/bash

root_path=$(pwd)
for i in nelastic_*; do
    cd ${i}
    for j in strain_*; do

        cd ${j}
        cp ${root_path}/POTCAR ./

        # IONIC position relax
        Myelas --vasp incar --encut 600.0 --isif 2 --kspace 0.10 -ctype rlx
        time mpirun -np 32 vasp_std >vasp_relax.log

        cp CONTCAR POSCAR

        # Static calculation
        Myelas --vasp incar --encut 600.0 --isif 2 --kspace 0.10 -ctype stc
        time mpirun -np 32 vasp_std >vasp_stc.log

        cd ..
    done
    cd ..
done
```





3. **Solve elastic constants**

```shell
Myelas -so 3D_2nd -smax 0.018 -snum 13 # The second elastic constants calculation
Myelas -so 3D_3rd -smax 0.060 -snum 13 # The third elastic constants calculation
```



**Results:**

**second_elastic.out:**

```txt
The cubic crystal mechanical properties
Cubic crystal (spacegroup No.: 227)

Elastic tensor C_ij (unit: GPa)
   153.056    57.116    57.116    0.000    0.000    0.000
   57.116    153.056    57.116    0.000    0.000    0.000
   57.116    57.116    153.056    0.000    0.000    0.000
   0.000    0.000    0.000    74.721    0.000    0.000
   0.000    0.000    0.000    0.000    74.721    0.000
   0.000    0.000    0.000    0.000    0.000    74.721

Compliance tensor S_ij (unit: GPa^-1)
   0.008196    -0.002227    -0.002227    0.000000    0.000000    0.000000
   -0.002227    0.008196    -0.002227    0.000000    0.000000    0.000000
   -0.002227    -0.002227    0.008196    0.000000    0.000000    0.000000
   0.000000    0.000000    0.000000    0.013383    0.000000    0.000000
   0.000000    0.000000    0.000000    0.000000    0.013383    0.000000
   0.000000    0.000000    0.000000    0.000000    0.000000    0.013383

mechanical stability:  Stable

unit cell volume :  163.5297 A^3
unit cell density:  2282.2924 kg/m^3

Polycrystalline modulus
(Unit: GPa) Bulk modulus    Shear modulus    Youngs modulus    Possion ratio    P-wave modulus
  Vogit      89.0962           64.0205           154.9485           0.2101           174.4568
  Reuss      89.0962           61.0930           149.1814           0.2209           170.5536
  Hill       89.0962           62.5568           152.0649           0.2155           172.5052

Cauchy Pressure  (GPa):  -17.6046
Pugh's ratio          :  0.7021
Vickers hardness (GPa):  11.8647

Anisotropy index:
  Zener anisotropy index        :  1.56
  Chung-Buessem anisotropy index:  0.02
  Universal anisotropy index    :  0.24
  Log-Euclidean anisotropy index:  0.10

Polycrystalline sound velocity (m/s)
  Longitudinal sound velocity:  8693.9170
  Shear sound velocity       :  5235.4196
  Bulk sound velocity        :  6248.0426
  Average sound velocity     :  5789.5759

Pure single-crystal sound velocity (m/s)
  [100] direction:  vl = 8189.156  vs1 = 5721.841  vs2 = 5721.841
  [110] direction:  vl = 8876.011  vs1 = 4584.559  vs2 = 5721.841
  [111] direction:  vl = 9093.440  vs1 = 4992.522  vs2 = 4992.522

Debye temperature:  630.48 K

The minimum thermal conductivity:
  Clark model  :  1.312 W/(m K)
  Chaill model :  1.427 W/(m K)
```



**ELADAT_3rd:**

```
The third elastic constants.
Please check the number of elastic constants for your structure.


C111 =  -746.5459
C112 =  -438.7388
C123 =  -92.4655


C144 =  59.9915
C166 =  -315.3537
C456 =  -71.2529

```



#### **5.1.2 Visualization**

**Input file:**

```
POSCAR-uc
POTCAR
second_elastics.out
```



**Commond:**

```shell
Myelas -p3D Youngs -ptype plane              #plot Young's modulus 
Myelas -p3D Shear -ptype plane -minmax max   #plot Max Shear modulus 
Myelas -p3D Shear -ptype plane -minmax min   #plot Max Shear modulus 
Myelas -p3D Poisson -ptype plane -minmax max #plot Max Shear modulus 
Myelas -p3D Poisson -ptype plane -minmax min #plot Max Shear modulus 
```



**Output file:**

![Young's modulus](F:\王豪-论文\写作\myelas\MyElas\example\bulk-materials-2nd-elastic-constants\Si\Young's modulus.png)

![Max poisson ratio](F:\王豪-论文\写作\myelas\MyElas\example\bulk-materials-2nd-elastic-constants\Si\Max poisson ratio.png)

![Min shear modulus](F:\王豪-论文\写作\myelas\MyElas\example\bulk-materials-2nd-elastic-constants\Si\Min shear modulus.png)

![](F:\王豪-论文\写作\myelas\MyElas\example\bulk-materials-2nd-elastic-constants\Si\Max poisson ratio.png)

![Min poisson ratio](F:\王豪-论文\写作\myelas\MyElas\example\bulk-materials-2nd-elastic-constants\Si\Min poisson ratio.png)

![Single Sound Velocity](F:\王豪-论文\写作\myelas\Single Sound Velocity.png)



<div STYLE="page-break-after: always;"></div>

### **5.2 2D-Graphene second elastic constants**     

#### **5.2.1 Elastic constants calculation**

**Input file:**

POSCAR-uc:

```
C2                                      
   1.00000000000000     
     2.4686779615000001    0.0000000000000000    0.0000000000000000
    -1.2343389807000000    2.1379378284000001    0.0000000000000000
     0.0000000000000000    0.0000000000000000   20.0000000000000000
   C 
     2
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.3333333429999996  0.6666666870000029  0.0000000000000000

```



1. **Generate the strain POSCAR:**

```shell
Myelas -g 2D_2nd -smax 0.018 -snum 13 # Only the second elastic constants
```



2. **DFT  calculation:**

Third elastic constants calculation example:

```shell
#!/bin/bash

root_path=$(pwd)
for i in 2D_nelastic_*; do
    cd ${i}
    for j in strain_*; do

        cd ${j}
        cp ${root_path}/POTCAR ./
        
        Myelas --vasp 2D_kpoints -kp 12 -kt G

        # IONIC position relax
        Myelas --vasp incar --encut 600.0 --isif 2 -ctype rlx
        time mpirun -np 32 vasp_std >vasp_relax.log

        cp CONTCAR POSCAR

        # Static calculation
        Myelas --vasp incar --encut 600.0 --isif 2 -ctype stc
        time mpirun -np 32 vasp_std >vasp_stc.log

        cd ..
    done
    cd ..
done
```



3. **Solve elastic constants**

```shell
Myelas -so 2D_2nd -smax 0.018 -snum 13 # The second elastic constants calculation
```



**Results:**

**second_elastic.out:**

```txt
2D elastic constants
Space group:  191

Elastic tensor C_ij (unit: N/m)
   353.612    60.918    0.000
   60.918    353.975    0.000
   0.000    0.000    146.877

Compliance tensor S_ij (unit: (N/m)^-1)
   0.002914    -0.000502    0.000000
   -0.000502    0.002911    0.000000
   0.000000    0.000000    0.006808

2D area (A^2) : 5.2779

Young(Ex and Ey) and shear(Gxy) moduli (unit: N/m)
Ex : 343.1280
Ey : 343.4804
Gxy: 146.8774

Poisson ratios(Muxy and Muyx)
Muxy : 0.1721
Muyx : 0.1723

mechanical stability:  Stable

Anisotropy index:
  Elastic anisotropy index      :  0.00
  Ranganathan anisotropy index  :  0.00
  Kube anisotropy index         :  0.00

```



#### **5.2.2 Visulization**

```
Myelas -p2D 2D
```



![2D Modulus](F:\王豪-论文\写作\myelas\Fig\2D Modulus.png)

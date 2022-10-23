import numpy as np
import os

from . import solve_elastic_3D

# import solve_elastic_3D


class read_elastic(object):
    def __init__(self):
        print("The software will directly read elastic tensor")

    def read_elas_tensor(self, elas_file=None, theta=None, phi=None, chi=None):

        elastic_tensor = np.zeros((6, 6))
        if elas_file == "OUTCAR":
            if os.path.isfile("OUTCAR"):
                print("OUTCAR file exists.")
                with open("OUTCAR", "r") as outcar:
                    text = outcar.readlines()
                    num_lines = len(text)

                    for n in np.arange(0, num_lines, 1):
                        if text[n] == " TOTAL ELASTIC MODULI (kBar)\n":

                            for index in np.arange(0, 6, 1):
                                elas_tensor = [
                                    float(elas)
                                    for elas in text[n + index + 3].split()[1:7]
                                ]

                                elastic_tensor[index, :] = np.array(elas_tensor) / 10.0

            else:
                print("Error! No OUTCAR file exists.")

        elif elas_file == "elastic.out":
            if os.path.isfile("elastic.out"):
                print("elastic.out file exists.")
                elas = np.loadtxt("elastic.out")
                for i in np.arange(0, 6, 1):
                    for j in np.arange(0, 6, 1):
                        elastic_tensor[i, j] = elas[i, j]
            else:
                print("Error! No OUTCAR file exists.")

        else:
            print("Error! Please give me OUTCAR or elastic.out.")

        # print(elastic_tensor)
        solve_elastic_3D.solve_elas3D().elasproperties(
            Celas=elastic_tensor, title="The elastic from OUTCAR or elastic.out", theta=theta, phi=phi
        )


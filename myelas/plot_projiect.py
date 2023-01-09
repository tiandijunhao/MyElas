from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import linecache
import os
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class plot_3D_modulus(object):
    def __init__(self, Celas=None, Temp=None):
        self.Temp = Temp
        if Celas == None:
            if os.path.isfile("second_elastic.out"):
                elasfile = linecache.getlines("second_elastic.out")
                elas = [line.strip() for line in elasfile]

                celas = np.zeros((6, 6))
                index = 0

                for c_elas in elas[4:10]:
                    c_elas = c_elas.split()
                    celas[index, :] = c_elas[0:6]
                    index += 1
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

                self.Celas = celas
            elif os.path.isfile("second_elastic_{}K.out".format(Temp)):
                elasfile = linecache.getlines("second_elastic_{}K.out".format(Temp))
                elas = [line.strip() for line in elasfile]

                celas = np.zeros((6, 6))
                index = 0

                for c_elas in elas[4:10]:
                    c_elas = c_elas.split()
                    celas[index, :] = c_elas[0:6]
                    index += 1
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

                self.Celas = celas
            else:
                print("Error!")
                print(
                    "Please input the second elastic tensor! You can give the second_elastic.out file!"
                )
        else:
            self.Celas = Celas

        self.S = np.linalg.inv(celas)

        self.m = 90
        self.n = 180
        self.o = 360
        self.phi = np.linspace(0, np.pi, self.m)  # z-xy angel
        self.theta = np.linspace(0, 2 * np.pi, self.n)  # x-y angel
        self.chi = np.linspace(0, 2 * np.pi, self.o)  # o=360

    def Youngs_3D_surf(self):
        E = np.matrix(np.zeros((self.m, self.n)))
        E_x = np.matrix(np.zeros((self.m, self.n)))
        E_y = np.matrix(np.zeros((self.m, self.n)))
        E_z = np.matrix(np.zeros((self.m, self.n)))

        U_V = np.matrix([[1], [1], [1], [0], [0], [0]])

        for i in np.arange(0, self.n, 1):
            for j in np.arange(0, self.m, 1):
                d = np.matrix(
                    [
                        [np.sin(self.phi[j]) * np.cos(self.theta[i])],
                        [np.sin(self.phi[j]) * np.sin(self.theta[i])],
                        [np.cos(self.phi[j])],
                    ]
                )

                D = d * d.T
                d_V = np.matrix(
                    [
                        [D[0, 0]],
                        [D[1, 1]],
                        [D[2, 2]],
                        [np.sqrt(2) * D[1, 2]],
                        [np.sqrt(2) * D[0, 2]],
                        [np.sqrt(2) * D[0, 1]],
                    ]
                )
                E[j, i] = np.linalg.inv(d_V.T * np.matrix(self.S) * d_V)
                # print(d_V)

                E_x[j, i] = E[j, i] * np.sin(self.phi[j]) * np.cos(self.theta[i])
                E_y[j, i] = E[j, i] * np.sin(self.phi[j]) * np.sin(self.theta[i])
                E_z[j, i] = E[j, i] * np.cos(self.phi[j])

        return E, E_x, E_y, E_z

    def Youngs_3D_plane(self):
        theta = np.linspace(0, 2 * np.pi, 10000)
        phi = np.linspace(0, 2 * np.pi, 10000)

        E_100 = np.zeros(10000)  # theta = pi/2
        E_010 = np.zeros(10000)  # theta = 0
        E_001 = np.zeros(10000)  # phi = pi/2

        x_100 = []
        y_100 = []
        x_010 = []
        y_010 = []
        x_001 = []
        y_001 = []

        # [100]
        for j in np.arange(0, 10000, 1):
            d1 = np.matrix(
                [
                    [np.sin(phi[j]) * np.cos(np.pi / 2)],
                    [np.sin(phi[j]) * np.sin(np.pi / 2)],
                    [np.cos(phi[j])],
                ]
            )

            D1 = d1 * d1.T
            d_V1 = np.matrix(
                [
                    [D1[0, 0]],
                    [D1[1, 1]],
                    [D1[2, 2]],
                    [np.sqrt(2) * D1[1, 2]],
                    [np.sqrt(2) * D1[0, 2]],
                    [np.sqrt(2) * D1[0, 1]],
                ]
            )
            E_100[j] = 1 / (d_V1.T * (self.S * d_V1))
            x_100.append(E_100[j] * np.sin(phi[j]))
            y_100.append(E_100[j] * np.cos(phi[j]))

            d2 = np.matrix(
                [
                    [np.sin(phi[j]) * np.cos(0)],
                    [np.sin(phi[j]) * np.sin(0)],
                    [np.cos(phi[j])],
                ]
            )

            D2 = d2 * d2.T
            d_V2 = np.matrix(
                [
                    [D2[0, 0]],
                    [D2[1, 1]],
                    [D2[2, 2]],
                    [np.sqrt(2) * D2[1, 2]],
                    [np.sqrt(2) * D2[0, 2]],
                    [np.sqrt(2) * D2[0, 1]],
                ]
            )
            E_010[j] = 1 / (d_V2.T * (self.S * d_V2))
            x_010.append(E_010[j] * np.sin(phi[j]))
            y_010.append(E_010[j] * np.cos(phi[j]))

            d3 = np.matrix(
                [
                    [np.sin(np.pi / 2) * np.cos(theta[j])],
                    [np.sin(np.pi / 2) * np.sin(theta[j])],
                    [np.cos(np.pi / 2)],
                ]
            )

            D3 = d3 * d3.T
            d_V3 = np.matrix(
                [
                    [D3[0, 0]],
                    [D3[1, 1]],
                    [D3[2, 2]],
                    [np.sqrt(2) * D3[1, 2]],
                    [np.sqrt(2) * D3[0, 2]],
                    [np.sqrt(2) * D3[0, 1]],
                ]
            )
            E_001[j] = 1 / (d_V3.T * (self.S * d_V3))
            x_001.append(E_001[j] * np.sin(theta[j]))
            y_001.append(E_001[j] * np.cos(theta[j]))
        Youngs_plane = [x_100, y_100, x_010, y_010, x_001, y_001]
        return Youngs_plane

    def Bulk_3D_surf(self):
        E = np.matrix(np.zeros((self.m, self.n)))
        E_x = np.matrix(np.zeros((self.m, self.n)))
        E_y = np.matrix(np.zeros((self.m, self.n)))
        E_z = np.matrix(np.zeros((self.m, self.n)))

        U_V = np.matrix([[1], [1], [1], [0], [0], [0]])

        for i in np.arange(0, self.n, 1):
            for j in np.arange(0, self.m, 1):
                d = np.matrix(
                    [
                        [np.sin(self.phi[j]) * np.cos(self.theta[i])],
                        [np.sin(self.phi[j]) * np.sin(self.theta[i])],
                        [np.cos(self.phi[j])],
                    ]
                )

                D = d * d.T
                d_V = np.matrix(
                    [
                        [D[0, 0]],
                        [D[1, 1]],
                        [D[2, 2]],
                        [np.sqrt(2) * D[1, 2]],
                        [np.sqrt(2) * D[0, 2]],
                        [np.sqrt(2) * D[0, 1]],
                    ]
                )

                # Bulk modulus
                E[j, i] = 1 / (3 * U_V.T * self.S * d_V)

                E_x[j, i] = E[j, i] * np.sin(self.phi[j]) * np.cos(self.theta[i])
                E_y[j, i] = E[j, i] * np.sin(self.phi[j]) * np.sin(self.theta[i])
                E_z[j, i] = E[j, i] * np.cos(self.phi[j])

        return E, E_x, E_y, E_z

    def Bulk_3D_plane(self):
        theta = np.linspace(0, 2 * np.pi, 10000)
        phi = np.linspace(0, 2 * np.pi, 10000)

        E_100 = np.zeros(10000)  # theta = pi/2
        E_010 = np.zeros(10000)  # theta = 0
        E_001 = np.zeros(10000)  # phi = pi/2

        x_100 = []
        y_100 = []
        x_010 = []
        y_010 = []
        x_001 = []
        y_001 = []

        U_V = np.matrix([[1], [1], [1], [0], [0], [0]])

        # [100]
        for j in np.arange(0, 10000, 1):
            d1 = np.matrix(
                [
                    [np.sin(phi[j]) * np.cos(np.pi / 2)],
                    [np.sin(phi[j]) * np.sin(np.pi / 2)],
                    [np.cos(phi[j])],
                ]
            )

            D1 = d1 * d1.T
            d_V1 = np.matrix(
                [
                    [D1[0, 0]],
                    [D1[1, 1]],
                    [D1[2, 2]],
                    [np.sqrt(2) * D1[1, 2]],
                    [np.sqrt(2) * D1[0, 2]],
                    [np.sqrt(2) * D1[0, 1]],
                ]
            )
            E_100[j] = 1 / (3 * U_V.T * (self.S * d_V1))
            x_100.append(E_100[j] * np.sin(phi[j]))
            y_100.append(E_100[j] * np.cos(phi[j]))


            d2 = np.matrix(
                [
                    [np.sin(phi[j]) * np.cos(0)],
                    [np.sin(phi[j]) * np.sin(0)],
                    [np.cos(phi[j])],
                ]
            )

            D2 = d2 * d2.T
            d_V2 = np.matrix(
                [
                    [D2[0, 0]],
                    [D2[1, 1]],
                    [D2[2, 2]],
                    [np.sqrt(2) * D2[1, 2]],
                    [np.sqrt(2) * D2[0, 2]],
                    [np.sqrt(2) * D2[0, 1]],
                ]
            )
            E_010[j] = 1 / (3 * U_V.T * (self.S * d_V2))
            x_010.append(E_010[j] * np.sin(phi[j]))
            y_010.append(E_010[j] * np.cos(phi[j]))


            d3 = np.matrix(
                [
                    [np.sin(np.pi / 2) * np.cos(theta[j])],
                    [np.sin(np.pi / 2) * np.sin(theta[j])],
                    [np.cos(np.pi / 2)],
                ]
            )

            D3 = d3 * d3.T
            d_V3 = np.matrix(
                [
                    [D3[0, 0]],
                    [D3[1, 1]],
                    [D3[2, 2]],
                    [np.sqrt(2) * D3[1, 2]],
                    [np.sqrt(2) * D3[0, 2]],
                    [np.sqrt(2) * D3[0, 1]],
                ]
            )
            E_001[j] = 1 / (3 * U_V.T * (self.S * d_V3))
            x_001.append(E_001[j] * np.sin(theta[j]))
            y_001.append(E_001[j] * np.cos(theta[j]))
        Bulk_plane = [x_100, y_100, x_010, y_010, x_001, y_001]
        return Bulk_plane

    def Shear_3D_surf(self, minmax=None):
        E = np.matrix(np.zeros((self.m, self.n)))
        E_x = np.matrix(np.zeros((self.m, self.n)))
        E_y = np.matrix(np.zeros((self.m, self.n)))
        E_z = np.matrix(np.zeros((self.m, self.n)))
        p_o = np.mat(np.zeros((3, self.o)))
        po = np.mat(np.zeros((1, self.o)))
        Go = np.mat(np.zeros((1, self.o)))

        U_V = np.matrix([[1], [1], [1], [0], [0], [0]])

        for i in np.arange(0, self.n, 1):
            for j in np.arange(0, self.m, 1):
                d = np.matrix(
                    [
                        [np.sin(self.phi[j]) * np.cos(self.theta[i])],
                        [np.sin(self.phi[j]) * np.sin(self.theta[i])],
                        [np.cos(self.phi[j])],
                    ]
                )

                D = d * d.T
                d_V = np.matrix(
                    [
                        [D[0, 0]],
                        [D[1, 1]],
                        [D[2, 2]],
                        [np.sqrt(2) * D[1, 2]],
                        [np.sqrt(2) * D[0, 2]],
                        [np.sqrt(2) * D[0, 1]],
                    ]
                )

                # Shear modulus
                for k in np.arange(0, self.o, 1):
                    p_o[0, k] = -np.cos(self.phi[j]) * np.cos(self.theta[i]) * np.cos(
                        self.chi[k]
                    ) + np.sin(self.theta[i]) * np.sin(self.chi[k])
                    p_o[1, k] = -np.cos(self.phi[j]) * np.sin(self.theta[i]) * np.cos(
                        self.chi[k]
                    ) - np.cos(self.theta[i]) * np.sin(self.chi[k])
                    p_o[2, k] = np.sin(self.phi[j]) * np.cos(self.chi[k])

                    N = p_o[:, k] * p_o[:, k].T
                    # p_V = np.matrix(
                    # [
                    #    [N[0, 0]],
                    #    [N[1, 1]],
                    #    [N[2, 2]],
                    #    [np.sqrt(2) * N[1, 2]],
                    #    [np.sqrt(2) * N[0, 2]],
                    #    [np.sqrt(2) * N[0, 1]],
                    # ]
                    # )

                    M = np.sqrt(2) * 0.5 * (d * p_o[:, k].T + p_o[:, k] * d.T)
                    m_V = np.matrix(
                        [
                            [M[0, 0]],
                            [M[1, 1]],
                            [M[2, 2]],
                            [np.sqrt(2) * M[1, 2]],
                            [np.sqrt(2) * M[0, 2]],
                            [np.sqrt(2) * M[0, 1]],
                        ]
                    )
                    Go[0, k] = 1 / (2 * m_V.T * self.S * m_V)

                if minmax == "max":
                    E[j, i] = Go.max()
                elif minmax == "min":
                    E[j, i] = Go.min()

                E_x[j, i] = E[j, i] * np.sin(self.phi[j]) * np.cos(self.theta[i])
                E_y[j, i] = E[j, i] * np.sin(self.phi[j]) * np.sin(self.theta[i])
                E_z[j, i] = E[j, i] * np.cos(self.phi[j])

        return E, E_x, E_y, E_z

    def Shear_3D_plane(self, minmax=None):
        theta = np.linspace(0, 2 * np.pi, 10000)
        phi = np.linspace(0, 2 * np.pi, 10000)

        E_100 = np.zeros(10000)  # theta = pi/2
        E_010 = np.zeros(10000)  # theta = 0
        E_001 = np.zeros(10000)  # phi = pi/2

        p_o1 = np.mat(np.zeros((3, self.o)))
        po1 = np.mat(np.zeros((1, self.o)))
        p_o2 = np.mat(np.zeros((3, self.o)))
        po2 = np.mat(np.zeros((1, self.o)))
        p_o3 = np.mat(np.zeros((3, self.o)))
        po3 = np.mat(np.zeros((1, self.o)))
        Go1 = np.mat(np.zeros((1, self.o)))
        Go2 = np.mat(np.zeros((1, self.o)))
        Go3 = np.mat(np.zeros((1, self.o)))

        x_100 = []
        y_100 = []
        x_010 = []
        y_010 = []
        x_001 = []
        y_001 = []

        U_V = np.matrix([[1], [1], [1], [0], [0], [0]])

        # [100]
        for j in np.arange(0, 10000, 1):
            d1 = np.matrix(
                [
                    [np.sin(phi[j]) * np.cos(np.pi / 2)],
                    [np.sin(phi[j]) * np.sin(np.pi / 2)],
                    [np.cos(phi[j])],
                ]
            )

            D1 = d1 * d1.T
            d_V1 = np.matrix(
                [
                    [D1[0, 0]],
                    [D1[1, 1]],
                    [D1[2, 2]],
                    [np.sqrt(2) * D1[1, 2]],
                    [np.sqrt(2) * D1[0, 2]],
                    [np.sqrt(2) * D1[0, 1]],
                ]
            )

            d2 = np.matrix(
                [
                    [np.sin(phi[j]) * np.cos(0)],
                    [np.sin(phi[j]) * np.sin(0)],
                    [np.cos(phi[j])],
                ]
            )

            D2 = d2 * d2.T
            d_V2 = np.matrix(
                [
                    [D2[0, 0]],
                    [D2[1, 1]],
                    [D2[2, 2]],
                    [np.sqrt(2) * D2[1, 2]],
                    [np.sqrt(2) * D2[0, 2]],
                    [np.sqrt(2) * D2[0, 1]],
                ]
            )

            d3 = np.matrix(
                [
                    [np.sin(np.pi / 2) * np.cos(theta[j])],
                    [np.sin(np.pi / 2) * np.sin(theta[j])],
                    [np.cos(np.pi / 2)],
                ]
            )

            D3 = d3 * d3.T
            d_V3 = np.matrix(
                [
                    [D3[0, 0]],
                    [D3[1, 1]],
                    [D3[2, 2]],
                    [np.sqrt(2) * D3[1, 2]],
                    [np.sqrt(2) * D3[0, 2]],
                    [np.sqrt(2) * D3[0, 1]],
                ]
            )

            for k in np.arange(0, self.o, 1):
                #100
                p_o1[0, k] = -np.cos(phi[j]) * np.cos(np.pi / 2) * np.cos(
                    self.chi[k]
                ) + np.sin(np.pi / 2) * np.sin(self.chi[k])
                p_o1[1, k] = -np.cos(phi[j]) * np.sin(np.pi / 2) * np.cos(
                    self.chi[k]
                ) - np.cos(np.pi / 2) * np.sin(self.chi[k])
                p_o1[2, k] = np.sin(phi[j]) * np.cos(self.chi[k])

                M1 = np.sqrt(2) * 0.5 * (d1 * p_o1[:, k].T + p_o1[:, k] * d1.T)
                m_V1 = np.matrix(
                    [
                        [M1[0, 0]],
                        [M1[1, 1]],
                        [M1[2, 2]],
                        [np.sqrt(2) * M1[1, 2]],
                        [np.sqrt(2) * M1[0, 2]],
                        [np.sqrt(2) * M1[0, 1]],
                    ]
                )
                Go1[0, k] = 1 / (2 * m_V1.T * self.S * m_V1)

                #010
                p_o2[0, k] = -np.cos(phi[j]) * np.cos(0) * np.cos(self.chi[k]) + np.sin(
                    0
                ) * np.sin(self.chi[k])
                p_o2[1, k] = -np.cos(phi[j]) * np.sin(0) * np.cos(self.chi[k]) - np.cos(
                    0
                ) * np.sin(self.chi[k])
                p_o2[2, k] = np.sin(phi[j]) * np.cos(self.chi[k])

                M2 = np.sqrt(2) * 0.5 * (d2 * p_o2[:, k].T + p_o2[:, k] * d2.T)
                m_V2 = np.matrix(
                    [
                        [M2[0, 0]],
                        [M2[1, 1]],
                        [M2[2, 2]],
                        [np.sqrt(2) * M2[1, 2]],
                        [np.sqrt(2) * M2[0, 2]],
                        [np.sqrt(2) * M2[0, 1]],
                    ]
                )
                Go2[0, k] = 1 / (2 * m_V2.T * self.S * m_V2)

                #001
                p_o3[0, k] = -np.cos(np.pi / 2) * np.cos(theta[j]) * np.cos(
                    self.chi[k]
                ) + np.sin(theta[j]) * np.sin(self.chi[k])
                p_o3[1, k] = -np.cos(np.pi / 2) * np.sin(theta[j]) * np.cos(
                    self.chi[k]
                ) - np.cos(theta[j]) * np.sin(self.chi[k])
                p_o3[2, k] = np.sin(np.pi / 2) * np.cos(self.chi[k])

                M3 = np.sqrt(2) * 0.5 * (d3 * p_o3[:, k].T + p_o3[:, k] * d3.T)
                m_V3 = np.matrix(
                    [
                        [M3[0, 0]],
                        [M3[1, 1]],
                        [M3[2, 2]],
                        [np.sqrt(2) * M3[1, 2]],
                        [np.sqrt(2) * M3[0, 2]],
                        [np.sqrt(2) * M3[0, 1]],
                    ]
                )
                Go3[0, k] = 1 / (2 * m_V3.T * self.S * m_V3)

            if minmax == "max":
                E_100[j] = Go1.max()
            elif minmax == "min":
                E_100[j] = Go1.min()
            x_100.append(E_100[j] * np.sin(phi[j]))
            y_100.append(E_100[j] * np.cos(phi[j]))

            if minmax == "max":
                E_010[j] = Go2.max()
            elif minmax == "min":
                E_010[j] = Go2.min()
            x_010.append(E_010[j] * np.sin(phi[j]))
            y_010.append(E_010[j] * np.cos(phi[j]))

            if minmax == "max":
                E_001[j] = Go3.max()
            elif minmax == "min":
                E_001[j] = Go3.min()
            x_001.append(E_001[j] * np.sin(theta[j]))
            y_001.append(E_001[j] * np.cos(theta[j]))
                
        Shear_plane = [x_100, y_100, x_010, y_010, x_001, y_001]
        return Shear_plane

    def Poisson_3D_surf(self, minmax=None):
        E = np.matrix(np.zeros((self.m, self.n)))
        E_x = np.matrix(np.zeros((self.m, self.n)))
        E_y = np.matrix(np.zeros((self.m, self.n)))
        E_z = np.matrix(np.zeros((self.m, self.n)))
        p_o = np.mat(np.zeros((3, self.o)))
        po = np.mat(np.zeros((1, self.o)))
        Go = np.mat(np.zeros((1, self.o)))

        U_V = np.matrix([[1], [1], [1], [0], [0], [0]])

        for i in np.arange(0, self.n, 1):
            for j in np.arange(0, self.m, 1):
                d = np.matrix(
                    [
                        [np.sin(self.phi[j]) * np.cos(self.theta[i])],
                        [np.sin(self.phi[j]) * np.sin(self.theta[i])],
                        [np.cos(self.phi[j])],
                    ]
                )

                D = d * d.T
                d_V = np.matrix(
                    [
                        [D[0, 0]],
                        [D[1, 1]],
                        [D[2, 2]],
                        [np.sqrt(2) * D[1, 2]],
                        [np.sqrt(2) * D[0, 2]],
                        [np.sqrt(2) * D[0, 1]],
                    ]
                )

                # Shear modulus
                for k in np.arange(0, self.o, 1):
                    p_o[0, k] = -np.cos(self.phi[j]) * np.cos(self.theta[i]) * np.cos(
                        self.chi[k]
                    ) + np.sin(self.theta[i]) * np.sin(self.chi[k])
                    p_o[1, k] = -np.cos(self.phi[j]) * np.sin(self.theta[i]) * np.cos(
                        self.chi[k]
                    ) - np.cos(self.theta[i]) * np.sin(self.chi[k])
                    p_o[2, k] = np.sin(self.phi[j]) * np.cos(self.chi[k])

                    N = p_o[:, k] * p_o[:, k].T
                    p_V = np.matrix(
                        [
                            [N[0, 0]],
                            [N[1, 1]],
                            [N[2, 2]],
                            [np.sqrt(2) * N[1, 2]],
                            [np.sqrt(2) * N[0, 2]],
                            [np.sqrt(2) * N[0, 1]],
                        ]
                    )

                    # M = np.sqrt(2) * 0.5 * (d * p_o[:, k].T + p_o[:, k] * d.T)
                    # m_V = np.matrix(
                    #    [
                    #        [M[0, 0]],
                    #        [M[1, 1]],
                    #        [M[2, 2]],
                    #        [np.sqrt(2) * M[1, 2]],
                    #        [np.sqrt(2) * M[0, 2]],
                    #        [np.sqrt(2) * M[0, 1]],
                    #    ]
                    # )
                    po[0, k] = (
                        -1 * (1 / (d_V.T * (self.S * d_V))) * d_V.T * (self.S * p_V)
                    )

                if minmax == "max":
                    E[j, i] = po.max()
                elif minmax == "min":
                    E[j, i] = po.min()

                E_x[j, i] = E[j, i] * np.sin(self.phi[j]) * np.cos(self.theta[i])
                E_y[j, i] = E[j, i] * np.sin(self.phi[j]) * np.sin(self.theta[i])
                E_z[j, i] = E[j, i] * np.cos(self.phi[j])

        return E, E_x, E_y, E_z

    def Poisson_3D_plane(self, minmax=None):
        theta = np.linspace(0, 2 * np.pi, 10000)
        phi = np.linspace(0, 2 * np.pi, 10000)

        E_100 = np.zeros(10000)  # theta = pi/2
        E_010 = np.zeros(10000)  # theta = 0
        E_001 = np.zeros(10000)  # phi = pi/2

        p_o1 = np.mat(np.zeros((3, self.o)))
        po1 = np.mat(np.zeros((1, self.o)))
        p_o2 = np.mat(np.zeros((3, self.o)))
        po2 = np.mat(np.zeros((1, self.o)))
        p_o3 = np.mat(np.zeros((3, self.o)))
        po3 = np.mat(np.zeros((1, self.o)))
        Go1 = np.mat(np.zeros((1, self.o)))
        Go2 = np.mat(np.zeros((1, self.o)))
        Go3 = np.mat(np.zeros((1, self.o)))

        x_100 = []
        y_100 = []
        x_010 = []
        y_010 = []
        x_001 = []
        y_001 = []

        U_V = np.matrix([[1], [1], [1], [0], [0], [0]])

        # [100]
        for j in np.arange(0, 10000, 1):
            d1 = np.matrix(
                [
                    [np.sin(phi[j]) * np.cos(np.pi / 2)],
                    [np.sin(phi[j]) * np.sin(np.pi / 2)],
                    [np.cos(phi[j])],
                ]
            )

            D1 = d1 * d1.T
            d_V1 = np.matrix(
                [
                    [D1[0, 0]],
                    [D1[1, 1]],
                    [D1[2, 2]],
                    [np.sqrt(2) * D1[1, 2]],
                    [np.sqrt(2) * D1[0, 2]],
                    [np.sqrt(2) * D1[0, 1]],
                ]
            )

            d2 = np.matrix(
                [
                    [np.sin(phi[j]) * np.cos(0)],
                    [np.sin(phi[j]) * np.sin(0)],
                    [np.cos(phi[j])],
                ]
            )

            D2 = d2 * d2.T
            d_V2 = np.matrix(
                [
                    [D2[0, 0]],
                    [D2[1, 1]],
                    [D2[2, 2]],
                    [np.sqrt(2) * D2[1, 2]],
                    [np.sqrt(2) * D2[0, 2]],
                    [np.sqrt(2) * D2[0, 1]],
                ]
            )

            d3 = np.matrix(
                [
                    [np.sin(np.pi / 2) * np.cos(theta[j])],
                    [np.sin(np.pi / 2) * np.sin(theta[j])],
                    [np.cos(np.pi / 2)],
                ]
            )

            D3 = d3 * d3.T
            d_V3 = np.matrix(
                [
                    [D3[0, 0]],
                    [D3[1, 1]],
                    [D3[2, 2]],
                    [np.sqrt(2) * D3[1, 2]],
                    [np.sqrt(2) * D3[0, 2]],
                    [np.sqrt(2) * D3[0, 1]],
                ]
            )
            
            for k in np.arange(0, self.o, 1):
                p_o1[0, k] = -np.cos(phi[j]) * np.cos(np.pi / 2) * np.cos(
                    self.chi[k]
                ) + np.sin(np.pi / 2) * np.sin(self.chi[k])
                p_o1[1, k] = -np.cos(phi[j]) * np.sin(np.pi / 2) * np.cos(
                    self.chi[k]
                ) - np.cos(np.pi / 2) * np.sin(self.chi[k])
                p_o1[2, k] = np.sin(phi[j]) * np.cos(self.chi[k])

                N1 = p_o1[:, k] * p_o1[:, k].T
                p_V1 = np.matrix(
                    [
                        [N1[0, 0]],
                        [N1[1, 1]],
                        [N1[2, 2]],
                        [np.sqrt(2) * N1[1, 2]],
                        [np.sqrt(2) * N1[0, 2]],
                        [np.sqrt(2) * N1[0, 1]],
                    ]
                )

                po1[0, k] = -1 * (1 / (d_V1.T * (self.S * d_V1))) * d_V1.T * (self.S * p_V1)

                #010
                p_o2[0, k] = -np.cos(phi[j]) * np.cos(0) * np.cos(self.chi[k]) + np.sin(
                    0
                ) * np.sin(self.chi[k])
                p_o2[1, k] = -np.cos(phi[j]) * np.sin(0) * np.cos(self.chi[k]) - np.cos(
                    0
                ) * np.sin(self.chi[k])
                p_o2[2, k] = np.sin(phi[j]) * np.cos(self.chi[k])

                N2 = p_o2[:, k] * p_o2[:, k].T
                p_V2 = np.matrix(
                    [
                        [N2[0, 0]],
                        [N2[1, 1]],
                        [N2[2, 2]],
                        [np.sqrt(2) * N2[1, 2]],
                        [np.sqrt(2) * N2[0, 2]],
                        [np.sqrt(2) * N2[0, 1]],
                    ]
                )

                po2[0, k] = -1 * (1 / (d_V2.T * (self.S * d_V2))) * d_V2.T * (self.S * p_V2)

                #001
                p_o3[0, k] = -np.cos(np.pi / 2) * np.cos(theta[j]) * np.cos(
                    self.chi[k]
                ) + np.sin(theta[j]) * np.sin(self.chi[k])
                p_o3[1, k] = -np.cos(np.pi / 2) * np.sin(theta[j]) * np.cos(
                    self.chi[k]
                ) - np.cos(theta[j]) * np.sin(self.chi[k])
                p_o3[2, k] = np.sin(np.pi / 2) * np.cos(self.chi[k])

                N3 = p_o3[:, k] * p_o3[:, k].T
                p_V3 = np.matrix(
                    [
                        [N3[0, 0]],
                        [N3[1, 1]],
                        [N3[2, 2]],
                        [np.sqrt(2) * N3[1, 2]],
                        [np.sqrt(2) * N3[0, 2]],
                        [np.sqrt(2) * N3[0, 1]],
                    ]
                )

                po3[0, k] = -1 * (1 / (d_V3.T * (self.S * d_V3))) * d_V3.T * (self.S * p_V3)

            if minmax == "max":
                E_100[j] = po1.max()
            elif minmax == "min":
                E_100[j] = po1.min()
            x_100.append(E_100[j] * np.sin(phi[j]))
            y_100.append(E_100[j] * np.cos(phi[j]))

            if minmax == "max":
                E_010[j] = po2.max()
            elif minmax == "min":
                E_010[j] = po2.min()
            x_010.append(E_010[j] * np.sin(phi[j]))
            y_010.append(E_010[j] * np.cos(phi[j]))
            
            if minmax == "max":
                E_001[j] = po3.max()
            elif minmax == "min":
                E_001[j] = po3.min()
            x_001.append(E_001[j] * np.sin(theta[j]))
            y_001.append(E_001[j] * np.cos(theta[j]))
   
        Poisson_plane = [x_100, y_100, x_010, y_010, x_001, y_001]
        return Poisson_plane

    def plot_3D_modulus(self, modulus_name=None, plot_type=None, minmax=None):
        """
        Plot single-crystal modulus in 3D 
        ------

        """

        E = np.matrix(np.zeros((self.m, self.n)))
        E_x = np.matrix(np.zeros((self.m, self.n)))
        E_y = np.matrix(np.zeros((self.m, self.n)))
        E_z = np.matrix(np.zeros((self.m, self.n)))
        E_max = 0
        E_min = 0
        Y_plane = []
        Title = []

        ## define the modulus which need plot
        if modulus_name == "Youngs":
            E, E_x, E_y, E_z = self.Youngs_3D_surf()
            Title.append("Young's modulus")
            E_max = E.max()
            E_min = E.min()

            if plot_type == "plane":
                Y_plane = self.Youngs_3D_plane()

        elif modulus_name == "Bulk":
            E, E_x, E_y, E_z = self.Bulk_3D_surf()
            Title.append("Bulk modulus")
            E_max = E.max()
            E_min = E.min()

            if plot_type == "plane":
                Y_plane = self.Bulk_3D_plane()

        elif modulus_name == "Shear":
            E, E_x, E_y, E_z = self.Shear_3D_surf(minmax=minmax)

            if minmax == "max":
                Title.append("Max shear modulus")
            elif minmax == "min":
                Title.append("Min shear modulus")

            E_max = E.max()
            E_min = E.min()

            if plot_type == "plane":
                Y_plane = self.Shear_3D_plane(minmax=minmax)

        elif modulus_name == "Poisson":
            E, E_x, E_y, E_z = self.Poisson_3D_surf(minmax=minmax)

            if minmax == "max":
                Title.append("Max poisson ratio")
            elif minmax == "min":
                Title.append("Min poisson ratio")

            E_max = E.max()
            E_min = E.min()

            if plot_type == "plane":
                Y_plane = self.Poisson_3D_plane(minmax=minmax)

        if abs(E_max - E_min) < 10 ** (-8):
            E_max = E_max + 0.05 * E_max
            E_min = E_min - 0.05 * E_min

        A=abs((E_max - E_min)/E_min*100) # Anisotropy
        ## Plot
        if plot_type == "3D":

            fig = plt.figure(figsize=[4, 3])
            ax = fig.add_subplot(111, projection="3d")

            plt.title(Title[0], fontsize=18)

            N_E = (E - E_min) / (E_max - E_min)
            surf = ax.plot_surface(
                E_x,
                E_y,
                E_z,
                rstride=1,
                cstride=1,
                facecolors=plt.cm.viridis(N_E),
                linewidth=0,
                antialiased=False,
                shade=False,
            )
            plt.xlabel("x")
            plt.ylabel("y")
            ax.set_zlabel("z")
            plt.xlim(-E_max - 0.1 * E_max, E_max + 0.1 * E_max)
            plt.ylim(-E_max - 0.1 * E_max, E_max + 0.1 * E_max)
            ax.set_zlim(-E_max - 0.1 * E_max, E_max + 0.1 * E_max)
            arrow_prop_dict = dict(
                mutation_scale=10, arrowstyle="->", shrinkA=0, shrinkB=0
            )

            a = Arrow3D(
                [-E_max - 0.3 * E_max, E_max + 0.3 * E_max],
                [0, 0],
                [0, 0],
                **arrow_prop_dict,
                color="black"
            )
            ax.add_artist(a)
            a = Arrow3D(
                [0, 0],
                [+E_max + 0.3 * E_max, -E_max - 0.3 * E_max],
                [0, 0],
                **arrow_prop_dict,
                color="black"
            )
            ax.add_artist(a)
            a = Arrow3D(
                [0, 0],
                [0, 0],
                [-E_max - 0.3 * E_max, E_max + 0.3 * E_max],
                **arrow_prop_dict,
                color="black"
            )
            ax.add_artist(a)

            # ax.text(0.0, 0.0, -0.1, r"$o$")
            ax.text(E_max + 0.3 * E_max, 0, 0, r"$x$")
            ax.text(0, -E_max - 0.35 * E_max, 0, r"$y$")
            ax.text(0, 0, E_max + 0.35 * E_max, r"$z$")

            cm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
            cm.set_array(E)
            cb = plt.colorbar(cm, fraction=0.046, pad=0.04, shrink=0.5, aspect=10)
            cb.set_ticks([E_min, E_max])
            cb.set_ticklabels(["{:.2f} GPa".format(E_min), "{:.2f} GPa".format(E_max)])
            cb.ax.tick_params(labelsize=16)
            ax.set_axis_off()

            ax.text((E_max+E_min)/2,(E_max+E_min)/2,0,'Anisotropy: {:.1f} %'.format(A), transform=ax.transAxes,fontsize=18)
            if self.Temp == None:
                plt.savefig(Title[0], dpi=600, bbox_inches="tight")
            else:
                plt.savefig(Title[0]+" {}".format(self.Temp), dpi=600, bbox_inches="tight")

        elif plot_type == "plane":

            fig = plt.figure(figsize=[17, 4])
            ax1 = fig.add_subplot(141, projection="3d")

            plt.title(Title[0], fontsize=20)

            N_E = (E - E_min) / (E_max - E_min)
            surf = ax1.plot_surface(
                E_x,
                E_y,
                E_z,
                rstride=1,
                cstride=1,
                facecolors=plt.cm.viridis(N_E),
                linewidth=0,
                antialiased=False,
                shade=False,
            )
            plt.xlabel("x")
            plt.ylabel("y")
            ax1.set_zlabel("z")
            plt.xlim(-E_max - 0.1 * E_max, E_max + 0.1 * E_max)
            plt.ylim(-E_max - 0.1 * E_max, E_max + 0.1 * E_max)
            ax1.set_zlim(-E_max - 0.1 * E_max, E_max + 0.1 * E_max)
            arrow_prop_dict = dict(
                mutation_scale=10, arrowstyle="->", shrinkA=0, shrinkB=0
            )

            a = Arrow3D(
                [-E_max - 0.3 * E_max, E_max + 0.3 * E_max],
                [0, 0],
                [0, 0],
                **arrow_prop_dict,
                color="black"
            )
            ax1.add_artist(a)
            a = Arrow3D(
                [0, 0],
                [+E_max + 0.3 * E_max, -E_max - 0.3 * E_max],
                [0, 0],
                **arrow_prop_dict,
                color="black"
            )
            ax1.add_artist(a)
            a = Arrow3D(
                [0, 0],
                [0, 0],
                [-E_max - 0.3 * E_max, E_max + 0.3 * E_max],
                **arrow_prop_dict,
                color="black"
            )
            ax1.add_artist(a)

            # ax1.text(0.0, 0.0, -0.1, r"$o$")
            ax1.text(E_max + 0.35 * E_max, 0, 0, r"$x$")
            ax1.text(0, -E_max - 0.35 * E_max, 0, r"$y$")
            ax1.text(0, 0, E_max + 0.35 * E_max, r"$z$")

            cm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
            cm.set_array(E)
            cb = plt.colorbar(cm, fraction=0.046, pad=0.001, shrink=0.6, aspect=10, orientation='horizontal')
            cb.set_ticks([E_min, E_max])
            # cb.set_ticklabels(["{:.2f} GPa".format(E_min), "{:.2f} GPa".format(E_max)])
            if modulus_name == "Poisson":
                cb.set_ticklabels(["{:.2f}".format(E_min), "{:.2f}".format(E_max)])
            else:
                cb.set_ticklabels(
                    ["{:.2f} GPa".format(E_min), "{:.2f} GPa".format(E_max)]
                )
            cb.ax.tick_params(labelsize=16)
            
            ax1.text((E_max+E_min)/2,(E_max+E_min)/2,0,'Anisotropy: {:.1f} %'.format(A), transform=ax1.transAxes,fontsize=18)
            ax1.set_axis_off()

            # plane

            ax2 = fig.add_subplot(142)

            ax2.plot(Y_plane[4], Y_plane[5], "b", label="[001] plane")
            plt.legend(fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            ax3 = fig.add_subplot(143)

            ax3.plot(Y_plane[2], Y_plane[3], "b", label="[010] plane")
            plt.legend(fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            ax4 = fig.add_subplot(144)

            ax4.plot(Y_plane[0], Y_plane[1], "b", label="[100] plane")
            plt.legend(fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            #plt.tight_layout()
            plt.subplots_adjust(wspace =0.25, hspace =0)
            if self.Temp == None:
                plt.savefig(Title[0], dpi=600, bbox_inches="tight")
            else:
                plt.savefig(Title[0]+" {}".format(self.Temp), dpi=600, bbox_inches="tight")


class plot_2D_modulus(object):
    def __init__(self, Celas=None):
        if Celas == None:
            if os.path.isfile("second_elastic.out"):
                elasfile = linecache.getlines("second_elastic.out")
                elas = [line.strip() for line in elasfile]

                celas = np.zeros((3, 3))
                index = 0

                for c_elas in elas[4:7]:
                    C_elas = c_elas.split()
                    celas[index, :] = C_elas[0:3]
                    index += 1

                self.Celas = celas

            else:
                print("Error!")
                print(
                    "Please input the second elastic tensor! You can give the second_elastic.out file!"
                )
        else:
            self.Celas = Celas

        self.n = 3600
        self.theta = np.linspace(0, 2 * np.pi, self.n)  # x-y angel

    def modulus_2D(self):
        C = self.Celas
        Theta = self.theta
        E_2D = []
        P_2D = []
        for i in np.arange(0, len(self.theta)):
            A0 = C[0, 0] * C[1, 1] - C[0, 1] ** 2
            E = A0 / (
                C[0, 0] * np.sin(Theta[i]) ** 4
                + C[1, 1] * np.cos(Theta[i]) ** 4
                + ((A0 / C[2, 2]) - 2 * C[0, 1])
                * (np.sin(Theta[i]) * np.cos(Theta[i])) ** 2
            )
            E_2D.append(E)

            A1 = (C[0, 0] * C[1, 1] - C[0, 1] ** 2) / C[2, 2]
            c = np.cos(Theta[i])
            s = np.sin(Theta[i])
            P = (
                C[0, 1] * (c ** 4 + s ** 4) - (C[0, 0] + C[1, 1] - A1) * (c * s) ** 2
            ) / (
                C[0, 0] * s ** 4 + C[1, 1] * c ** 4 + (A1 - 2 * C[0, 1]) * (c * s) ** 2
            )
            P_2D.append(P)
        return E_2D, P_2D

    def plot_2D(self):
        E_2D, P_2D = self.modulus_2D()

        E_max = np.matrix(E_2D).max()
        E_min = np.matrix(E_2D).min()
        P_max = np.matrix(P_2D).max()
        P_min = np.matrix(P_2D).min()

        fig = plt.figure(figsize=[12, 6])

        ax1 = fig.add_subplot(121, projection="polar")
        ax1.plot(self.theta, E_2D, c="g", linewidth=3)
        ax1.set_rlim(E_min - 0.5 * E_min, E_max + 0.2 * E_max)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # plt.grid(axis='x',linestyle='--')
        plt.title("Young's modulus (N/m)", fontsize=18)

        ax2 = fig.add_subplot(122, projection="polar")
        ax2.plot(self.theta, P_2D, c="g", linewidth=3)
        ax2.set_rlim(P_min - 0.5 * P_min, P_max + 0.2 * P_max)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title("Poisson ratio", fontsize=18)

        plt.tight_layout()
        plt.savefig("2D Modulus", dpi=600, bbox_inches="tight")


class plot_3D_sv(object):
    def __init__(self, Celas=None):
        if Celas == None:
            if os.path.isfile("second_elastic.out"):
                elasfile = linecache.getlines("second_elastic.out")
                elas = [line.strip() for line in elasfile]

                celas = np.zeros((6, 6))
                index = 0

                for c_elas in elas[4:10]:
                    c_elas = c_elas.split()
                    celas[index, :] = c_elas[0:6]
                    index += 1

                self.Celas = celas
                self.density = elas[22].split()[3]
                print(self.density)
            else:
                print("Error!")
                print(
                    "Please input the second elastic tensor! You can give the second_elastic.out file!"
                )
        else:
            self.Celas = Celas

        self.S = np.linalg.inv(celas)
        self.m = 180
        self.n = 360

    def calc_single_sound_velocity(self, theta, phi, type=None):
        """
        To calculate the single crystal sound velocity

        """
        density = float(self.density)
        Celas = self.Celas

        n1 = np.sin(phi) * np.cos(theta)
        n2 = np.sin(phi) * np.sin(theta)
        n3 = np.cos(phi)
        Fv = np.zeros((3, 3))
        Fv[0, 0] = (
            n1 * n1 * Celas[0, 0]
            + n1 * n2 * Celas[0, 5]
            + n1 * n3 * Celas[0, 4]
            + n2 * n1 * Celas[5, 0]
            + n2 * n2 * Celas[5, 5]
            + n2 * n3 * Celas[5, 4]
            + n3 * n1 * Celas[4, 0]
            + n3 * n2 * Celas[4, 5]
            + n3 * n3 * Celas[4, 4]
        )
        Fv[1, 1] = (
            n1 * n1 * Celas[5, 5]
            + n1 * n2 * Celas[5, 1]
            + n1 * n3 * Celas[5, 3]
            + n2 * n1 * Celas[1, 5]
            + n2 * n2 * Celas[1, 1]
            + n2 * n3 * Celas[1, 3]
            + n3 * n1 * Celas[3, 5]
            + n3 * n2 * Celas[3, 1]
            + n3 * n3 * Celas[3, 3]
        )
        Fv[2, 2] = (
            n1 * n1 * Celas[4, 4]
            + n1 * n2 * Celas[4, 3]
            + n1 * n3 * Celas[4, 2]
            + n2 * n1 * Celas[3, 4]
            + n2 * n2 * Celas[3, 3]
            + n2 * n3 * Celas[3, 2]
            + n3 * n1 * Celas[2, 4]
            + n3 * n2 * Celas[2, 3]
            + n3 * n3 * Celas[2, 2]
        )
        Fv[0, 1] = Fv[1, 0] = (
            n1 * n1 * Celas[0, 5]
            + n1 * n2 * Celas[0, 1]
            + n1 * n3 * Celas[0, 3]
            + n2 * n1 * Celas[5, 5]
            + n2 * n2 * Celas[5, 1]
            + n2 * n3 * Celas[5, 3]
            + n3 * n1 * Celas[4, 5]
            + n3 * n2 * Celas[4, 1]
            + n3 * n3 * Celas[4, 3]
        )
        Fv[0, 2] = Fv[2, 0] = (
            n1 * n1 * Celas[0, 4]
            + n1 * n2 * Celas[0, 3]
            + n1 * n3 * Celas[0, 2]
            + n2 * n1 * Celas[5, 4]
            + n2 * n2 * Celas[5, 3]
            + n2 * n3 * Celas[5, 2]
            + n3 * n1 * Celas[4, 4]
            + n3 * n2 * Celas[4, 3]
            + n3 * n3 * Celas[4, 2]
        )
        Fv[1, 2] = Fv[2, 1] = (
            n1 * n1 * Celas[5, 4]
            + n1 * n2 * Celas[5, 3]
            + n1 * n3 * Celas[5, 2]
            + n2 * n1 * Celas[1, 4]
            + n2 * n2 * Celas[1, 3]
            + n2 * n3 * Celas[1, 2]
            + n3 * n1 * Celas[3, 4]
            + n3 * n2 * Celas[3, 3]
            + n3 * n3 * Celas[3, 2]
        )
        V_s, V_stensor = np.linalg.eigh(Fv * (10 ** 9))
        v_s = []
        for k in np.arange(0, len(V_s), 1):
            v = np.sqrt((V_s[k] / density)) / 1000.00
            v_s.append(v)
        VL = v_s[2]
        VS1 = v_s[0]
        VS2 = v_s[1]

        VL_x = VL * n1
        VL_y = VL * n2
        VL_z = VL * n3
        VS1_x = VS1 * n1
        VS1_y = VS1 * n2
        VS1_z = VS1 * n3
        VS2_x = VS2 * n1
        VS2_y = VS2 * n2
        VS2_z = VS2 * n3

        if type == "primary":
            return VL
        elif type == "fast":
            return VS2
        elif type == "slow":
            return VS1
        elif type == "all":
            return (
                VL,
                VS1,
                VS2,
                VL_x,
                VL_y,
                VL_z,
                VS1_x,
                VS1_y,
                VS1_z,
                VS2_x,
                VS2_y,
                VS2_z,
            )

    def plot_3d_sv(self, E=None, E_x=None, E_y=None, E_z=None, ax1=None, title=None,colorbar_direcrion=None):
        ax1.set_title(title, fontsize=16)

        E_min = E.min()
        E_max = E.max()
        A = abs(200 * (E_max - E_min) / (E_max + E_min))  # seismic anisotropy

        N_E = (E - E_min) / (E_max - E_min)
        surf = ax1.plot_surface(
            E_x,
            E_y,
            E_z,
            rstride=1,
            cstride=1,
            facecolors=plt.cm.viridis(N_E),
            linewidth=0,
            antialiased=False,
            shade=False,
        )
        plt.xlabel("x")
        plt.ylabel("y")
        ax1.set_zlabel("z")
        plt.xlim(-E_max - 0.01 * E_max, E_max + 0.01 * E_max)
        plt.ylim(-E_max - 0.01 * E_max, E_max + 0.01 * E_max)
        ax1.set_zlim(-E_max - 0.01 * E_max, E_max + 0.01 * E_max)
        arrow_prop_dict = dict(mutation_scale=10, arrowstyle="->", shrinkA=0, shrinkB=0)
        a = Arrow3D(
            [-E_max - 0.3 * E_max, E_max + 0.3 * E_max],
            [0, 0],
            [0, 0],
            **arrow_prop_dict,
            color="black"
        )
        ax1.add_artist(a)
        a = Arrow3D(
            [0, 0],
            [+E_max + 0.3 * E_max, -E_max - 0.3 * E_max],
            [0, 0],
            **arrow_prop_dict,
            color="black"
        )
        ax1.add_artist(a)
        a = Arrow3D(
            [0, 0],
            [0, 0],
            [-E_max - 0.3 * E_max, E_max + 0.3 * E_max],
            **arrow_prop_dict,
            color="black"
        )
        ax1.add_artist(a)

        ax1.text(
            (E_max + E_min) / 2,
            (E_max + E_min) / 2,
            0,
            "Seismic anisotropy: {:.1f} %".format(A),
            transform=ax1.transAxes,
            fontsize=16,
        )
        # ax1.text(0.0, 0.0, -0.1, r"$o$")
        ax1.text(E_max + 0.35 * E_max, 0, 0, r"$x$")
        ax1.text(0, -E_max - 0.35 * E_max, 0, r"$y$")
        ax1.text(0, 0, E_max + 0.35 * E_max, r"$z$")
        cm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        cm.set_array(E)
        if colorbar_direcrion == 'h':
            cb = plt.colorbar(cm, fraction=0.046, pad=0.04, shrink=0.5, aspect=10,orientation='horizontal')
        else:
            cb = plt.colorbar(cm, fraction=0.046, pad=0.04, shrink=0.5, aspect=10)
        cb.set_ticks([E_min, E_max])
        # cb.set_ticklabels(["{:.2f} GPa".format(E_min), "{:.2f} GPa".format(E_max)])
        cb.set_ticklabels(["{:.2f} km/s".format(E_min), "{:.2f} km/s".format(E_max)])
        cb.ax.tick_params(labelsize=16)
        ax1.set_axis_off()

    def plot_sv(self, plot_type="3D", type="all"):
        m = self.m
        n = self.n
        o = 360
        phi = np.linspace(0, np.pi, m)  # z-xy angel
        theta = np.linspace(0, 2 * np.pi, n)  # x-y angel
        chi = np.linspace(0, 2 * np.pi, o)  # o=360
        
        theta1 = np.linspace(-np.pi / 4, np.pi / 4, 1000)
        phi1 = np.linspace(np.pi / 2, 54 * np.pi / 180, 1000)
        phi2 = np.linspace(np.pi / 2, 0, 1000)
        
        VL = np.matrix(np.zeros((self.m, self.n)))
        VL_x = np.matrix(np.zeros((self.m, self.n)))
        VL_y = np.matrix(np.zeros((self.m, self.n)))
        VL_z = np.matrix(np.zeros((self.m, self.n)))
        
        VS1 = np.matrix(np.zeros((self.m, self.n)))
        VS1_x = np.matrix(np.zeros((self.m, self.n)))
        VS1_y = np.matrix(np.zeros((self.m, self.n)))
        VS1_z = np.matrix(np.zeros((self.m, self.n)))
        
        VS2 = np.matrix(np.zeros((self.m, self.n)))
        VS2_x = np.matrix(np.zeros((self.m, self.n)))
        VS2_y = np.matrix(np.zeros((self.m, self.n)))
        VS2_z = np.matrix(np.zeros((self.m, self.n)))
        if plot_type == "3D":
            for i in np.arange(0, self.n, 1):
                for j in np.arange(0, self.m, 1):
                    (
                        VL[j, i],
                        VS1[j, i],
                        VS2[j, i],
                        VL_x[j, i],
                        VL_y[j, i],
                        VL_z[j, i],
                        VS1_x[j, i],
                        VS1_y[j, i],
                        VS1_z[j, i],
                        VS2_x[j, i],
                        VS2_y[j, i],
                        VS2_z[j, i],
                    ) = self.calc_single_sound_velocity(
                        theta=theta[i], phi=phi[j], type='all'
                    )

            fig = plt.figure(figsize=[15, 6])
            ax1 = fig.add_subplot(131, projection="3d")
            self.plot_3d_sv(
                E=VS1, E_x=VS1_x, E_y=VS1_y, E_z=VS1_z, ax1=ax1, title="Slow Secondary", colorbar_direcrion='h'
            )

            ax2 = fig.add_subplot(132, projection="3d")
            self.plot_3d_sv(
                E=VS2, E_x=VS2_x, E_y=VS2_y, E_z=VS2_z, ax1=ax2, title="Fast Secondary", colorbar_direcrion='h'
            )

            ax3 = fig.add_subplot(133, projection="3d")
            self.plot_3d_sv(E=VL, E_x=VL_x, E_y=VL_y, E_z=VL_z, ax1=ax3, title="Primary", colorbar_direcrion='h')
            plt.subplots_adjust(wspace=0.20)
            if self.Temp == None:
                plt.savefig("Single Sound Velocity.png", dpi=600, bbox_inches="tight")
            else:
                plt.savefig("Single Sound Velocity_{}K.png".format(self.Temp), dpi=600, bbox_inches="tight")
            
        elif plot_type =="plane":
            V_001=[]
            V_010=[]
            V_100 = []
            x_100 = []
            y_100 = []
            x_010 = []
            y_010 = []
            x_001 = []
            y_001 = []
            for i in np.arange(0, self.n, 1):
                    for j in np.arange(0, self.m, 1):
                        (
                            VL[j, i],
                            VS1[j, i],
                            VS2[j, i],
                            VL_x[j, i],
                            VL_y[j, i],
                            VL_z[j, i],
                            VS1_x[j, i],
                            VS1_y[j, i],
                            VS1_z[j, i],
                            VS2_x[j, i],
                            VS2_y[j, i],
                            VS2_z[j, i],
                        ) = self.calc_single_sound_velocity(
                            theta=theta[i], phi=phi[j], type='all'
                        )
            
            for m in np.arange(0,1000,1):
                V_001.append(self.calc_single_sound_velocity(theta=theta1[m], phi=np.pi/2,type=type))
                V_010.append(self.calc_single_sound_velocity(theta=0, phi=phi2[m],type=type))
                V_100.append(self.calc_single_sound_velocity(theta=np.pi/2, phi=phi2[m],type=type))
                
            V_001 = V_001 + V_001[::-1] + V_001 + V_001[::-1]
            V_010 = V_010 + V_010[::-1] + V_010 + V_010[::-1]
            V_100 = V_100 + V_100[::-1] + V_100 + V_100[::-1]
            
            thet = np.linspace(0, 2 * np.pi, len(V_100))
            for n in np.arange(0, len(V_100), 1):
                x_100.append(V_100[n] * np.cos(thet[n]))
                y_100.append(V_100[n] * np.sin(thet[n]))
                x_010.append(V_010[n] * np.cos(thet[n]))
                y_010.append(V_010[n] * np.sin(thet[n]))
                x_001.append(V_001[n] * np.cos(thet[n]))
                y_001.append(V_001[n] * np.sin(thet[n]))

            fig = plt.figure(figsize=[17, 4])
            if type == 'primary':
                ax1 = fig.add_subplot(141, projection="3d")
                self.plot_3d_sv(E=VL, E_x=VL_x, E_y=VL_y, E_z=VL_z, ax1=ax1, title="Primary", colorbar_direcrion='h')
                
                ax2 = fig.add_subplot(142)

                ax2.plot(x_100, y_100, "b", label="[100] plane")
                plt.legend(fontsize=20)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                
                ax3 = fig.add_subplot(143)

                ax3.plot(x_010, y_010, "b", label="[010] plane")
                plt.legend(fontsize=20)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                
                ax4 = fig.add_subplot(144)

                ax4.plot(x_001, y_001, "b", label="[001] plane")
                plt.legend(fontsize=20)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                
                plt.subplots_adjust(wspace=0.25, hspace=0)
                if self.Temp == None:
                    plt.savefig('primary_inplane', dpi=600, bbox_inches="tight")
                else:
                    plt.savefig('primary_inplane_{}K'.format(self.Temp), dpi=600, bbox_inches="tight")
                
            elif type == 'fast':
                ax1 = fig.add_subplot(141, projection="3d")
                self.plot_3d_sv(E=VS2, E_x=VS2_x, E_y=VS2_y, E_z=VS2_z, ax1=ax1, title="Fast Secondary", colorbar_direcrion='h')
                
                ax2 = fig.add_subplot(142)

                ax2.plot(x_100, y_100, "b", label="[100] plane")
                plt.legend(fontsize=20)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                
                ax3 = fig.add_subplot(143)

                ax3.plot(x_010, y_010, "b", label="[010] plane")
                plt.legend(fontsize=20)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                
                ax4 = fig.add_subplot(144)

                ax4.plot(x_001, y_001, "b", label="[001] plane")
                plt.legend(fontsize=20)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                
                plt.subplots_adjust(wspace=0.25, hspace=0)
                if self.Temp == None:
                    plt.savefig('fast_inplane', dpi=600, bbox_inches="tight")
                else:
                    plt.savefig('fast_inplane_{}K'.format(self.Temp), dpi=600, bbox_inches="tight")
                
            elif type == 'slow':
                ax1 = fig.add_subplot(141, projection="3d")
                self.plot_3d_sv(E=VS1, E_x=VS1_x, E_y=VS1_y, E_z=VS1_z, ax1=ax1, title="Slow Secondary", colorbar_direcrion='h')
                
                ax2 = fig.add_subplot(142)

                ax2.plot(x_100, y_100, "b", label="[100] plane")
                plt.legend(fontsize=20)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                
                ax3 = fig.add_subplot(143)

                ax3.plot(x_010, y_010, "b", label="[010] plane")
                plt.legend(fontsize=20)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                
                ax4 = fig.add_subplot(144)

                ax4.plot(x_001, y_001, "b", label="[001] plane")
                plt.legend(fontsize=20)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                
                plt.subplots_adjust(wspace=0.25, hspace=0)
                if self.Temp == None:
                    plt.savefig('slow_inplane', dpi=600, bbox_inches="tight")
                else:
                    plt.savefig('slow_inplane_{}K'.format(self.Temp), dpi=600, bbox_inches="tight")
                
# plot_3D_sv().calc_single_sound_elocity()
# plot_3D_modulus().plot_3D_modulus(modulus_name="Bulk", plot_type="3D", minmax="3D")
# plot_2D_modulus().plot_2D()


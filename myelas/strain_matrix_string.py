import os


def Elastics_3D(spg_num=None):
    if spg_num >= 1 and spg_num <= 2:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, x, 0, 0, 0, 0]",
            "[0, 0, x, 0, 0, 0]",
            "[0, 0, 0, x, 0, 0]",
            "[0, 0, 0, 0, x, 0]",
            "[0, 0, 0, 0, 0, x]",
            "[x, x, 0, 0, 0, 0]",
            "[x, 0, x, 0, 0, 0]",
            "[x, 0, 0, x, 0, 0]",
            "[x, 0, 0, 0, x, 0]",
            "[x, 0, 0, 0, 0, x]",
            "[0, x, x, 0, 0, 0]",
            "[0, x, 0, x, 0, 0]",
            "[0, x, 0, 0, x, 0]",
            "[0, x, 0, 0, 0, x]",
            "[0, 0, x, x, 0, 0]",
            "[0, 0, x, 0, x, 0]",
            "[0, 0, x, 0, 0, x]",
            "[0, 0, 0, x, x, 0]",
            "[0, 0, 0, x, 0, x]",
            "[0, 0, 0, 0, x, x]",
        ]

    elif spg_num >= 3 and spg_num <= 15:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, x, 0, 0, 0, 0]",
            "[0, 0, x, 0, 0, 0]",
            "[0, 0, 0, x, 0, 0]",
            "[0, 0, 0, 0, x, 0]",
            "[0, 0, 0, 0, 0, x]",
            "[x, x, 0, 0, 0, 0]",
            "[0, x, x, 0, 0, 0]",
            "[x, 0, x, 0, 0, 0]",
            "[x, 0, 0, 0, 0, x]",
            "[0, x, 0, 0, 0, x]",
            "[0, 0, x, 0, 0, x]",
            "[0, 0, 0, x, x, 0]",
        ]

    elif spg_num >= 16 and spg_num <= 74:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, x, 0, 0, 0, 0]",
            "[0, 0, x, 0, 0, 0]",
            "[0, 0, 0, x, 0, 0]",
            "[0, 0, 0, 0, x, 0]",
            "[0, 0, 0, 0, 0, x]",
            "[x, x, 0, 0, 0, 0]",
            "[0, x, x, 0, 0, 0]",
            "[x, 0, x, 0, 0, 0]",
        ]

    elif spg_num >= 75 and spg_num <= 88:
        strain_matrix = [
            "[x, x, 0, 0, 0, 0]",
            "[0, 0, 0, 0, 0, x]",
            "[0, 0, x, 0, 0, 0]",
            "[0, 0, 0, x, x, 0]",
            "[x, x, x, 0, 0, 0]",
            "[0, x, x, 0, 0, 0]",
            "[x, 0, 0, 0, 0, x]",
        ]

    elif spg_num >= 89 and spg_num <= 142:
        strain_matrix = [
            "[x, x, 0, 0, 0, 0]",
            "[0, 0, 0, 0, 0, x]",
            "[0, 0, x, 0, 0, 0]",
            "[0, 0, 0, x, x, 0]",
            "[x, x, x, 0, 0, 0]",
            "[0, x, x, 0, 0, 0]",
        ]

    elif spg_num >= 143 and spg_num <= 148:
        strain_matrix = [
            "[x, x, 0, 0, 0, 0]",
            "[0, 0, 0, 0, 0, x]",
            "[0, 0, x, 0, 0, 0]",
            "[0, 0, 0, x, x, 0]",
            "[x, x, x, 0, 0, 0]",
            "[0, 0, 0, 0, x, x]",
            "[0, x, 0, 0, 0, x]",
            "[0, 0, 0, x, 0, x]",
        ]

    elif spg_num >= 149 and spg_num <= 167:
        strain_matrix = [
            "[x, x, 0, 0, 0, 0]",
            "[0, 0, 0, 0, 0, x]",
            "[0, 0, x, 0, 0, 0]",
            "[0, 0, 0, x, x, 0]",
            "[x, x, x, 0, 0, 0]",
            "[0, 0, 0, 0, x, x]",
        ]

    elif spg_num >= 168 and spg_num <= 194:
        strain_matrix = [
            "[x, x, 0, 0, 0, 0]",
            "[0, 0, 0, 0, 0, x]",
            "[0, 0, x, 0, 0, 0]",
            "[0, 0, 0, x, x, 0]",
            "[x, x, x, 0, 0, 0]",
        ]

    elif spg_num >= 195 and spg_num <= 230:
        strain_matrix = [
            "[0, 0, 0, x, x, x]",
            "[x, x, 0, 0, 0, 0]",
            "[x, x, x, 0, 0, 0]",
        ]

    return strain_matrix


def Elastics_3D3rd(nelastic=None):
    if nelastic == 14:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[x, x, 0, 0, 0, 0]",
            "[x, x, x, 0, 0, 0]",
            "[x, 0, 0, x, 0, 0]",
            "[0, x, 0, 0, 0, 0]",
            "[0, 0, x, 0, 0, 0]",
            "[0, x, x, 0, 0, 0]",
            "[0, 0, 0, x, 0, 0]",
            "[0, 0, x, 0, 0, x]",
            "[0, x, 0, x, 0, 0]",
            "[0, x, 0, 0, x, 0]",
            "[0, 0, x, 0, x, 0]",
            "[x, 0, x, x, 0, 0]",
            "[x, x, 0, 0, x, 0]",
        ]

    elif nelastic == 12:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, 0, x, 0, 0, 0]",
            "[x, -x, 0, 0, 0, 0]",
            "[x, 0, x, 0, 0, 0]",
            "[x, 0, -x, 0, 0, 0]",
            "[x, x, 0, 0, 0, 0]",
            "[0, 0, x, 0, 0, x]",
            "[x, 0, 0, x, 0, 0]",
            "[0, x, 0, x, 0, 0]",
            "[0, 0, x, x, 0, 0]",
            "[x, 0, 0, 0, 0, x]",
            "[x, 0, 0, x, x, 0]",
        ]

    elif nelastic == 10:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, 0, x, 0, 0, 0]",
            "[x, -x, 0, 0, 0, 0]",
            "[x, 0, x, 0, 0, 0]",
            "[x, 0, -x, 0, 0, 0]",
            "[x, x, 0, 0, 0, 0]",
            "[0, 0, x, 0, 0, x]",
            "[x, 0, 0, x, 0, 0]",
            "[0, x, 0, x, 0, 0]",
            "[0, 0, x, x, 0, 0]",
        ]

    elif nelastic == 8:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[x, x, 0, 0, 0, 0]",
            "[x, 0, x, 0, 0, 0]",
            "[x, x, x, 0, 0, 0]",
            "[x, 0, 0, x, 0, 0]",
            "[x, 0, 0, 0, x, 0]",
            "[x, 0, 0, 0, 0, x]",
            "[0, 0, 0, x, x, x]",
        ]

    elif nelastic == 6:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[x, x, 0, 0, 0, 0]",
            "[x, x, x, 0, 0, 0]",
            "[x, 0, 0, x, 0, 0]",
            "[x, 0, 0, 0, 0, x]",
            "[0, 0, 0, x, x, x]",
        ]

    return strain_matrix


def Elastics_2D(nelastic=None):
    if nelastic == 6:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, x, 0, 0, 0, 0]",
            "[0, 0, 0, 0, 0, x]",
            "[x, x, 0, 0, 0, 0]",
            "[x, 0, 0, 0, 0, x]",
            "[0, x, 0, 0, 0, x]",
        ]

    
    elif nelastic == 4:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, x, 0, 0, 0, 0]",
            "[0, 0, 0, 0, 0, x]",
            "[x, x, 0, 0, 0, 0]",
        ]
    
    else:
        print("error")

    return strain_matrix

def Elastics_NVT(spg_num=None):
    if spg_num >= 1 and spg_num <= 2:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, x, 0, 0, 0, 0]",
            "[0, 0, x, 0, 0, 0]",
            "[0, 0, 0, x, 0, 0]",
            "[0, 0, 0, 0, x, 0]",
            "[0, 0, 0, 0, 0, x]",
        ]

    elif spg_num >= 3 and spg_num <= 15:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, x, 0, 0, 0, 0]",
            "[0, 0, x, x, 0, 0]",
            "[0, 0, 0, 0, x, x]",
        ]

    elif spg_num >= 16 and spg_num <= 74:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, x, 0, 0, 0, 0]",
            "[0, 0, x, x, x, x]",
        ]

    elif spg_num >= 75 and spg_num <= 88:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, 0, x, x, 0, x]",
        ]

    elif spg_num >= 89 and spg_num <= 142:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, 0, x, x, 0, x]",
        ]

    elif spg_num >= 143 and spg_num <= 148:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, 0, x, x, 0, 0]",
        ]

    elif spg_num >= 149 and spg_num <= 167:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, 0, x, x, 0, 0]",
        ]

    elif spg_num >= 168 and spg_num <= 194:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, 0, x, x, 0, 0]",
        ]

    elif spg_num >= 195 and spg_num <= 230:
        strain_matrix = [
            "[x, 0, 0, 0, 0, 0]",
            "[0, 0, 0, x, 0, 0]",
        ]

    return strain_matrix
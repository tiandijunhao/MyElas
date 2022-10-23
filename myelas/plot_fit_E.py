import matplotlib.pyplot as plt
import numpy as np

class plot_energy_fit(object):
    def plot_energy(self, x=None, y=None, fit_func=None, figname=None, strain_str=None):
        plt.figure()
        plt.plot(x,y,'o',c='b')
        plt.plot(x, fit_func(x), c='r', linewidth = 2)
        plt.ylabel('(E-E0)/V0', fontsize = 16)
        plt.xlabel('Strain', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.title(figname, fontsize=18)
        plt.text(0.9*x[0],0.9*y[0],"strain = "+strain_str,fontsize=16)

        plt.savefig(figname+'.png', dpi=300, bbox_inches='tight')


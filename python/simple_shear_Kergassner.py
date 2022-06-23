import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

double_slip_lh_1_H_0 = np.loadtxt(
            "../results/simple_shear_Kergassner/double_slip_lh_1_H_0/stress12_vs_shear_strain_at_boundary.txt",
            delimiter="|", usecols = (1, 2), skiprows = 1)

double_slip_lh_2_H_0 = np.loadtxt(
            "../results/simple_shear_Kergassner/double_slip_lh_2_H_0/stress12_vs_shear_strain_at_boundary.txt",
            delimiter="|", usecols = (1, 2), skiprows = 1)

double_slip_lh_2e_1_H_0 = np.loadtxt(
            "../results/simple_shear_Kergassner/double_slip_lh_2e-1_H_0/stress12_vs_shear_strain_at_boundary.txt",
            delimiter="|", usecols = (1, 2), skiprows = 1)

fig, axs = plt.subplots(1,3)

axs[0].plot(double_slip_lh_2e_1_H_0[:,0], double_slip_lh_2e_1_H_0[:,1], label='l/h = 0.2, H = 0')
axs[0].plot(double_slip_lh_1_H_0[:,0], double_slip_lh_1_H_0[:,1], label='l/h = 1, H = 0')
axs[0].plot(double_slip_lh_2_H_0[:,0], double_slip_lh_2_H_0[:,1], label='l/h = 2, H = 0')
axs[0].set(xlabel = "gamma", ylabel = "sigma_12 / tau_ref")
axs[0].legend()

double_slip_lh_1_H_0 = np.loadtxt(
            "../results/simple_shear_Kergassner/double_slip_lh_1_H_0/plot_over_line.txt",
            delimiter=",", skiprows = 1)

double_slip_lh_2_H_0 = np.loadtxt(
            "../results/simple_shear_Kergassner/double_slip_lh_2_H_0/plot_over_line.txt",
            delimiter=",", skiprows = 1)

double_slip_lh_2e_1_H_0 = np.loadtxt(
            "../results/simple_shear_Kergassner/double_slip_lh_2e-1_H_0/plot_over_line.txt",
            delimiter=",", skiprows = 1)

double_slip_lh_2e_1_H_100 = np.loadtxt(
            "../results/simple_shear_Kergassner/double_slip_lh_2e-1_H_100/plot_over_line.txt",
            delimiter=",", skiprows = 1)

double_slip_lh_2e_1_H_1000 = np.loadtxt(
            "../results/simple_shear_Kergassner/double_slip_lh_2e-1_H_1000/plot_over_line.txt",
            delimiter=",", skiprows = 1)

axs[1].plot(np.abs(double_slip_lh_2e_1_H_0[:,4]), double_slip_lh_2e_1_H_0[:,9], label='l/h = 0.2, H = 0')
axs[1].plot(np.abs(double_slip_lh_1_H_0[:,4]), double_slip_lh_1_H_0[:,9], label='l/h = 1, H = 0')
axs[1].plot(np.abs(double_slip_lh_2_H_0[:,4]), double_slip_lh_2_H_0[:,9], label='l/h = 2, H = 0')
axs[1].set(xlabel = "abs(slip)", ylabel = "x2 / h")
axs[1].legend()

axs[2].plot(np.abs(double_slip_lh_2e_1_H_0[:,4]), double_slip_lh_2e_1_H_0[:,9], label='l/h = 0.2, H = 0')
axs[2].plot(np.abs(double_slip_lh_2e_1_H_100[:,4]), double_slip_lh_2e_1_H_100[:,9], label='l/h = 0.2, H = 100')
axs[2].plot(np.abs(double_slip_lh_2e_1_H_1000[:,4]), double_slip_lh_2e_1_H_1000[:,9], label='l/h = 0.2, H = 1000')
axs[2].set(xlabel = "abs(slip)", ylabel = "x2 / h")
axs[2].legend()

# Output plot
plt.show()

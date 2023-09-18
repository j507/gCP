import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

no_slips = np.loadtxt(
            "../results/simple_shear/no_slips/stress12_vs_shear_at_boundary.txt",
            delimiter="|", usecols = (1, 2), skiprows = 1)

single_slip_hl_125e0_Ht_0 = np.loadtxt(
            "../results/simple_shear/single_slip_hl_125e0_Ht_0/stress12_vs_shear_at_boundary.txt",
            delimiter="|", usecols = (1, 2), skiprows = 1)

single_slip_hl_125e_2_Ht_0 = np.loadtxt(
            "../results/simple_shear/single_slip_hl_125e-2_Ht_0/stress12_vs_shear_at_boundary.txt",
            delimiter="|", usecols = (1, 2), skiprows = 1)

double_slip_hl_125e_2_Ht_0 = np.loadtxt(
            "../results/simple_shear/double_slip_hl_125e-2_Ht_0/stress12_vs_shear_at_boundary.txt",
            delimiter="|", usecols = (1, 2), skiprows = 1)

double_slip_hl_125e_2_Ht_2e_1 = np.loadtxt(
            "../results/simple_shear/double_slip_hl_125e-2_Ht_2e-1/stress12_vs_shear_at_boundary.txt",
            delimiter="|", usecols = (1,  2), skiprows = 1)

double_slip_hl_125e0_Ht_0 = np.loadtxt(
            "../results/simple_shear/double_slip_hl_125e0_Ht_0/stress12_vs_shear_at_boundary.txt",
            delimiter="|", usecols = (1, 2), skiprows = 1)

double_slip_hl_125e0_Ht_2e_1 = np.loadtxt(
            "../results/simple_shear/double_slip_hl_125e0_Ht_2e-1/stress12_vs_shear_at_boundary.txt",
            delimiter="|", usecols = (1,  2), skiprows = 1)

fig, axs = plt.subplots(2,2)

axs[0,0].plot(no_slips[:,0], no_slips[:,1]/50, label='elastic')
axs[0,0].plot(single_slip_hl_125e0_Ht_0[:,0], single_slip_hl_125e0_Ht_0[:,1]/50, label='h/l = 125')
axs[0,0].plot(single_slip_hl_125e_2_Ht_0[:,0], single_slip_hl_125e_2_Ht_0[:,1]/50, label='h/l = 1.25')
axs[0,0].legend()
axs[0,0].set(xlabel = "gamma", ylabel = "sigma_12 / tau_ref")
axs[0,1].plot(double_slip_hl_125e_2_Ht_0[:,0], double_slip_hl_125e_2_Ht_0[:,1]/50, label='h/l = 1.25, H/tau = 0')
axs[0,1].plot(double_slip_hl_125e_2_Ht_2e_1[:,0], double_slip_hl_125e_2_Ht_2e_1[:,1]/50, label='h/l = 1.25, H/tau = 2e-1')
axs[0,1].plot(double_slip_hl_125e0_Ht_0[:,0], double_slip_hl_125e0_Ht_0[:,1]/50, label='h/l = 125, H/tau = 0')
axs[0,1].plot(double_slip_hl_125e0_Ht_2e_1[:,0], double_slip_hl_125e0_Ht_2e_1[:,1]/50, label='h/l = 125, H/tau = 2e-1')
axs[0,1].set(xlabel = "gamma", ylabel = "sigma_12 / tau_ref")
axs[0,1].legend()


double_slip_hl_125e_2_Ht_0 = np.loadtxt(
            "../results/simple_shear/double_slip_hl_125e-2_Ht_0/strain12_vs_height.txt",
            delimiter=",", usecols = (0, 7), skiprows = 1)

double_slip_hl_125e_2_Ht_2e_1 = np.loadtxt(
            "../results/simple_shear/double_slip_hl_125e-2_Ht_2e-1/strain12_vs_height.txt",
            delimiter=",", usecols = (0, 7), skiprows = 1)

double_slip_hl_125e0_Ht_0 = np.loadtxt(
            "../results/simple_shear/double_slip_hl_125e0_Ht_0/strain12_vs_height.txt",
            delimiter=",", usecols = (0, 7), skiprows = 1)

double_slip_hl_125e0_Ht_2e_1 = np.loadtxt(
            "../results/simple_shear/double_slip_hl_125e0_Ht_2e-1/strain12_vs_height.txt",
            delimiter=",", usecols = (0, 7), skiprows = 1)

axs[1,0].plot(double_slip_hl_125e0_Ht_0[:,0], double_slip_hl_125e0_Ht_0[:,1], label='h/l = 125, H/tau = 0.0')
axs[1,0].plot(double_slip_hl_125e_2_Ht_0[:,0], double_slip_hl_125e_2_Ht_0[:,1], label='h/l = 1.25, H/tau = 0.0')
axs[1,0].set(xlabel = "2epsilon_12", ylabel = "x2 / h")
axs[1,0].legend()
axs[1,1].plot(double_slip_hl_125e0_Ht_2e_1[:,0], double_slip_hl_125e0_Ht_2e_1[:,1], label='h/l = 125, H/tau = 2e-1')
axs[1,1].plot(double_slip_hl_125e_2_Ht_2e_1[:,0], double_slip_hl_125e_2_Ht_2e_1[:,1], label='h/l = 1.25, H/tau = 2e-1')
axs[1,1].set(xlabel = "2epsilon_12", ylabel = "x2 / h")
axs[1,1].legend()

# Output plot
plt.show()

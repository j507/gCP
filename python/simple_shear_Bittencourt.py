import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

no_slips = np.loadtxt(
            "../results/simple_shear_Bittencourt/no_slips/stress12_vs_shear_strain_at_boundary.txt",
            delimiter="|", usecols = (1, 2), skiprows = 1)

single_slip_hl_125e0_Ht_0 = np.loadtxt(
            "../results/simple_shear_Bittencourt/single_slip_hl_125e0_Ht_0/stress12_vs_shear_strain_at_boundary.txt",
            delimiter="|", usecols = (1, 2), skiprows = 1)

single_slip_hl_125e_2_Ht_0 = np.loadtxt(
            "../results/simple_shear_Bittencourt/single_slip_hl_125e-2_Ht_0/stress12_vs_shear_strain_at_boundary.txt",
            delimiter="|", usecols = (1, 2), skiprows = 1)

double_slip_hl_125e_2_Ht_0 = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_125e-2_Ht_0/stress12_vs_shear_strain_at_boundary.txt",
            delimiter="|", usecols = (1, 2), skiprows = 1)

double_slip_hl_125e_2_Ht_2e_1 = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_125e-2_Ht_2e-1/stress12_vs_shear_strain_at_boundary.txt",
            delimiter="|", usecols = (1,  2), skiprows = 1)

double_slip_hl_125e0_Ht_0 = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_125e0_Ht_0/stress12_vs_shear_strain_at_boundary.txt",
            delimiter="|", usecols = (1, 2), skiprows = 1)

double_slip_hl_125e0_Ht_2e_1 = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_125e0_Ht_2e-1/stress12_vs_shear_strain_at_boundary.txt",
            delimiter="|", usecols = (1,  2), skiprows = 1)

double_slip_hl_35e_1_Ht_2e0 = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_35e-1_Ht_2e0/stress12_vs_shear_strain_at_boundary.txt",
            delimiter="|", usecols = (1,  2), skiprows = 1)

fig, axs = plt.subplots(2,4)

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
axs[0,3].plot(double_slip_hl_35e_1_Ht_2e0[:,0], double_slip_hl_35e_1_Ht_2e0[:,1]/50, label='h/l = 3.5, H/tau = 2')
axs[0,3].set(xlabel = "gamma", ylabel = "sigma_12 / tau_ref")
axs[0,3].legend()

single_slip_hl_125e0_Ht_0 = np.loadtxt(
            "../results/simple_shear_Bittencourt/single_slip_hl_125e0_Ht_0/plot_over_line.txt",
            delimiter=",", skiprows = 1)
single_slip_hl_125e_2_Ht_0 = np.loadtxt(
            "../results/simple_shear_Bittencourt/single_slip_hl_125e-2_Ht_0/plot_over_line.txt",
            delimiter=",", skiprows = 1)
single_slip_hl_125e0_Ht_0_ = np.loadtxt(
            "../results/simple_shear_Bittencourt/single_slip_hl_125e0_Ht_0/plot_over_line_311.txt",
            delimiter=",", skiprows = 1)
single_slip_hl_125e_2_Ht_0_ = np.loadtxt(
            "../results/simple_shear_Bittencourt/single_slip_hl_125e-2_Ht_0/plot_over_line_311.txt",
            delimiter=",", skiprows = 1)
double_slip_hl_125e_2_Ht_0 = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_125e-2_Ht_0/plot_over_line.txt",
            delimiter=",", skiprows = 1)
double_slip_hl_125e0_Ht_0 = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_125e0_Ht_0/plot_over_line.txt",
            delimiter=",", skiprows = 1)
double_slip_hl_125e_2_Ht_2e_2 = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_125e-2_Ht_2e-2/plot_over_line.txt",
            delimiter=",", skiprows = 1)
double_slip_hl_125e0_Ht_2e_2 = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_125e0_Ht_2e-2/plot_over_line.txt",
            delimiter=",", skiprows = 1)
double_slip_hl_25_Ht_0 = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_25_Ht_0/plot_over_line.txt",
            delimiter=",", skiprows = 1)
double_slip_hl_25_Ht_2e_2 = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_25_Ht_2e-2/plot_over_line.txt",
            delimiter=",", skiprows = 1)
double_slip_hl_25_Ht_2e_1 = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_25_Ht_2e-1/plot_over_line.txt",
            delimiter=",", skiprows = 1)
double_slip_hl_25_Ht_2e0 = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_25_Ht_2e0/plot_over_line.txt",
            delimiter=",", skiprows = 1)
double_slip_hl_35e_1_Ht_2e0 = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_35e-1_Ht_2e0/plot_over_line.txt",
            delimiter=",", skiprows = 1)
double_slip_hl_35e_1_Ht_2e0_ = np.loadtxt(
            "../results/simple_shear_Bittencourt/double_slip_hl_35e-1_Ht_2e0/plot_over_line_311.txt",
            delimiter=",", skiprows = 1)

axs[1,0].plot(np.abs(single_slip_hl_125e_2_Ht_0[:,0]), single_slip_hl_125e_2_Ht_0[:,8], label='h/l = 1.25, H = 0, G = 0.0218')
axs[1,0].plot(np.abs(single_slip_hl_125e0_Ht_0[:,0]), single_slip_hl_125e0_Ht_0[:,8], label='h/l = 125, H = 0, G = 0.0218')
axs[1,0].plot(np.abs(single_slip_hl_125e_2_Ht_0_[:,0]), single_slip_hl_125e_2_Ht_0_[:,8], label='h/l = 1.25, H = 0, G = 0.0068016')
axs[1,0].plot(np.abs(single_slip_hl_125e0_Ht_0_[:,0]), single_slip_hl_125e0_Ht_0_[:,8], label='h/l = 125, H = 0, G = 0.0068016')
axs[1,0].set(xlabel = "2epsilon_12", ylabel = "x2 / h")
axs[1,0].set_xlim(0,0.025)
axs[1,0].legend()
axs[1,1].plot(np.abs(double_slip_hl_125e_2_Ht_0[:,0]), double_slip_hl_125e_2_Ht_0[:,9], label='h/l = 1.25, H = 0')
axs[1,1].plot(np.abs(double_slip_hl_125e0_Ht_0[:,0]), double_slip_hl_125e0_Ht_0[:,9], label='h/l = 125, H = 0')
axs[1,1].set(xlabel = "2epsilon_12", ylabel = "x2 / h")
axs[1,1].legend()
axs[0,2].plot(np.abs(double_slip_hl_125e_2_Ht_0[:,0]), double_slip_hl_125e_2_Ht_0[:,9], label='h/l = 1.25, H = 0.02')
axs[0,2].plot(np.abs(double_slip_hl_25_Ht_2e_2[:,0]), double_slip_hl_25_Ht_2e_2[:,9], label='h/l = 25, H = 0.02')
axs[0,2].plot(np.abs(double_slip_hl_125e0_Ht_0[:,0]), double_slip_hl_125e0_Ht_0[:,9], label='h/l = 125, H = 0.02')
axs[0,2].set(xlabel = "2epsilon_12", ylabel = "x2 / h")
axs[0,2].legend()
axs[1,2].plot(np.abs(double_slip_hl_25_Ht_0[:,0]), double_slip_hl_25_Ht_0[:,9], label='h/l = 25, H = 0')
axs[1,2].plot(np.abs(double_slip_hl_25_Ht_2e_2[:,0]), double_slip_hl_25_Ht_2e_2[:,9], label='h/l = 25, H = 0.02')
axs[1,2].plot(np.abs(double_slip_hl_25_Ht_2e_1[:,0]), double_slip_hl_25_Ht_2e_1[:,9], label='h/l = 25, H = 0.2')
axs[1,2].plot(np.abs(double_slip_hl_25_Ht_2e0[:,0]), double_slip_hl_25_Ht_2e0[:,9], label='h/l = 25, H = 2')
axs[1,2].set(xlabel = "2epsilon_12", ylabel = "x2 / h")
axs[1,2].legend()
axs[1,3].plot(np.abs(double_slip_hl_35e_1_Ht_2e0[:,0]), double_slip_hl_35e_1_Ht_2e0[:,9], label='h/l = 3.5, H = 2, G = 0.0218')
axs[1,3].plot(np.abs(double_slip_hl_35e_1_Ht_2e0_[:,0]), double_slip_hl_35e_1_Ht_2e0_[:,9], label='h/l = 3.5, H = 2, G = 0.0068016')
axs[1,3].set(xlabel = "2epsilon_12", ylabel = "x2 / h")
axs[1,3].legend()

# Output plot
plt.show()

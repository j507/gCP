import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


fig, axs = plt.subplots(1,3)

grain_boundaries_lh_5e0_lambda_0 = np.loadtxt(
            "../results/simple_shear_Kergassner/grain_boundaries_lh_5e0_lambda_0/slips_vs_height.txt",
            delimiter=",", skiprows = 1)

grain_boundaries_lh_5e0_lambda_1e4 = np.loadtxt(
            "../results/simple_shear_Kergassner/grain_boundaries_lh_5e0_lambda_1e4/slips_vs_height.txt",
            delimiter=",", skiprows = 1)

grain_boundaries_lh_2e_1 = np.loadtxt(
            "../results/simple_shear_Kergassner/grain_boundaries_lh_2e-1/slips_vs_height.txt",
            delimiter=",", skiprows = 1)

grain_boundaries_lh_2e0 = np.loadtxt(
            "../results/simple_shear_Kergassner/grain_boundaries_lh_2e0/slips_vs_height.txt",
            delimiter=",", skiprows = 1)

grain_boundaries_lh_2e0_alpha_0 = np.loadtxt(
            "../results/simple_shear_Kergassner/grain_boundaries_lh_2e0_alpha_0/slips_vs_height.txt",
            delimiter=",", skiprows = 1)

grain_boundaries_lh_2e0_alpha_20 = np.loadtxt(
            "../results/simple_shear_Kergassner/grain_boundaries_lh_2e0_alpha_20/slips_vs_height.txt",
            delimiter=",", skiprows = 1)

axs[0].plot(np.abs(grain_boundaries_lh_5e0_lambda_0[0:330,0]), grain_boundaries_lh_5e0_lambda_0[0:330,9], label='')
axs[0].plot(np.abs(grain_boundaries_lh_5e0_lambda_1e4[0:330,0]), grain_boundaries_lh_5e0_lambda_1e4[0:330,9], label='')
axs[0].plot(np.abs(grain_boundaries_lh_5e0_lambda_0[0:330,1]), grain_boundaries_lh_5e0_lambda_0[0:330,9], label='')
axs[0].plot(np.abs(grain_boundaries_lh_5e0_lambda_1e4[0:330,1]), grain_boundaries_lh_5e0_lambda_1e4[0:330,9], label='')
axs[0].plot(np.abs(grain_boundaries_lh_5e0_lambda_0[333:670,2]), grain_boundaries_lh_5e0_lambda_0[333:670,9], label='')
axs[0].plot(np.abs(grain_boundaries_lh_5e0_lambda_1e4[333:670,2]), grain_boundaries_lh_5e0_lambda_1e4[333:670,9], label='')
axs[0].plot(np.abs(grain_boundaries_lh_5e0_lambda_0[333:670,3]), grain_boundaries_lh_5e0_lambda_0[333:670,9], label='')
axs[0].plot(np.abs(grain_boundaries_lh_5e0_lambda_1e4[333:670,3]), grain_boundaries_lh_5e0_lambda_1e4[333:670,9], label='')
axs[0].plot(np.abs(grain_boundaries_lh_5e0_lambda_0[671:999,4]), grain_boundaries_lh_5e0_lambda_0[671:999,9], label='')
axs[0].plot(np.abs(grain_boundaries_lh_5e0_lambda_1e4[671:999,4]), grain_boundaries_lh_5e0_lambda_1e4[671:999,9], label='')
axs[0].plot(np.abs(grain_boundaries_lh_5e0_lambda_0[671:999,5]), grain_boundaries_lh_5e0_lambda_0[671:999,9], label='')
axs[0].plot(np.abs(grain_boundaries_lh_5e0_lambda_1e4[671:999,5]), grain_boundaries_lh_5e0_lambda_1e4[671:999,9], label='')
axs[0].set(xlabel = "abs(slip)", ylabel = "x2 / h")
axs[0].legend()

axs[2].plot(np.abs(grain_boundaries_lh_2e_1[0:330,0]), grain_boundaries_lh_2e_1[0:330,9], label='')
axs[2].plot(np.abs(grain_boundaries_lh_2e0[0:330,0]), grain_boundaries_lh_2e0[0:330,9], label='')
axs[2].plot(np.abs(grain_boundaries_lh_2e_1[0:330,1]), grain_boundaries_lh_2e_1[0:330,9], label='')
axs[2].plot(np.abs(grain_boundaries_lh_2e0[0:330,1]), grain_boundaries_lh_2e0[0:330,9], label='')
axs[2].plot(np.abs(grain_boundaries_lh_2e_1[333:670,2]), grain_boundaries_lh_2e_1[333:670,9], label='')
axs[2].plot(np.abs(grain_boundaries_lh_2e0[333:670,2]), grain_boundaries_lh_2e0[333:670,9], label='')
axs[2].plot(np.abs(grain_boundaries_lh_2e_1[333:670,3]), grain_boundaries_lh_2e_1[333:670,9], label='')
axs[2].plot(np.abs(grain_boundaries_lh_2e0[333:670,3]), grain_boundaries_lh_2e0[333:670,9], label='')
axs[2].plot(np.abs(grain_boundaries_lh_2e_1[671:999,4]), grain_boundaries_lh_2e_1[671:999,9], label='')
axs[2].plot(np.abs(grain_boundaries_lh_2e0[671:999,4]), grain_boundaries_lh_2e0[671:999,9], label='')
axs[2].plot(np.abs(grain_boundaries_lh_2e_1[671:999,5]), grain_boundaries_lh_2e_1[671:999,9], label='')
axs[2].plot(np.abs(grain_boundaries_lh_2e0[671:999,5]), grain_boundaries_lh_2e0[671:999,9], label='')
axs[2].set(xlabel = "abs(slip)", ylabel = "x2 / h")
axs[2].legend()

axs[1].plot(np.abs(grain_boundaries_lh_2e0_alpha_0[0:330,0]), grain_boundaries_lh_2e0_alpha_0[0:330,9], label='')
axs[1].plot(np.abs(grain_boundaries_lh_2e0_alpha_20[0:330,0]), grain_boundaries_lh_2e0_alpha_20[0:330,9], label='')
axs[1].plot(np.abs(grain_boundaries_lh_2e0_alpha_0[0:330,1]), grain_boundaries_lh_2e0_alpha_0[0:330,9], label='')
axs[1].plot(np.abs(grain_boundaries_lh_2e0_alpha_20[0:330,1]), grain_boundaries_lh_2e0_alpha_20[0:330,9], label='')
axs[1].plot(np.abs(grain_boundaries_lh_2e0_alpha_0[333:670,2]), grain_boundaries_lh_2e0_alpha_0[333:670,9], label='')
axs[1].plot(np.abs(grain_boundaries_lh_2e0_alpha_20[333:670,2]), grain_boundaries_lh_2e0_alpha_20[333:670,9], label='')
axs[1].plot(np.abs(grain_boundaries_lh_2e0_alpha_0[333:670,3]), grain_boundaries_lh_2e0_alpha_0[333:670,9], label='')
axs[1].plot(np.abs(grain_boundaries_lh_2e0_alpha_20[333:670,3]), grain_boundaries_lh_2e0_alpha_20[333:670,9], label='')
axs[1].plot(np.abs(grain_boundaries_lh_2e0_alpha_0[671:999,4]), grain_boundaries_lh_2e0_alpha_0[671:999,9], label='')
axs[1].plot(np.abs(grain_boundaries_lh_2e0_alpha_20[671:999,4]), grain_boundaries_lh_2e0_alpha_20[671:999,9], label='')
axs[1].plot(np.abs(grain_boundaries_lh_2e0_alpha_0[671:999,5]), grain_boundaries_lh_2e0_alpha_0[671:999,9], label='')
axs[1].plot(np.abs(grain_boundaries_lh_2e0_alpha_20[671:999,5]), grain_boundaries_lh_2e0_alpha_20[671:999,9], label='')
axs[1].set(xlabel = "abs(slip)", ylabel = "x2 / h")
axs[1].legend()

# Output plot
plt.show()

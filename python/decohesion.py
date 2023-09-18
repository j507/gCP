import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

data = np.loadtxt(
            "../results/simple_shear_default/decohesion_log.txt")


fig, axs = plt.subplots(2,2)

axs[0,0].plot(data[:,2], data[:,0], label='Traction')
axs[0,0].set(xlabel = "gamma", ylabel = "sigma_12 / tau_ref")
axs[0,0].legend()
axs[0,1].plot(data[:,2], data[:,1], label='damage')
axs[0,1].set(xlabel = "gamma", ylabel = "sigma_12 / tau_ref")
axs[0,1].legend()

# Output plot
plt.show()

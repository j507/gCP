import math
import numpy
import matplotlib.pyplot
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

D_250_15Ep3_1Em2_1_02 = \
  numpy.loadtxt(
    "/home/iwtm36/iwtm36/gCP/results/final_report/250_15Ep3_1Em2_1_02/homogenization.txt",
    delimiter="|", usecols = numpy.arange(1,30), skiprows = 1)

D_250_15Ep3_1Em2_1_04 = \
  numpy.loadtxt(
    "/home/iwtm36/iwtm36/gCP/results/final_report/250_15Ep3_1Em2_1_04/homogenization.txt",
    delimiter="|", usecols = numpy.arange(1,30), skiprows = 1)

D_250_15Ep3_1Em2_1_06 = \
  numpy.loadtxt(
    "/home/iwtm36/iwtm36/gCP/results/final_report/250_15Ep3_1Em2_1_06/homogenization.txt",
    delimiter="|", usecols = numpy.arange(1,30), skiprows = 1)

D_250_15Ep3_1Em2_1_08 = \
  numpy.loadtxt(
    "/home/iwtm36/iwtm36/gCP/results/final_report/250_15Ep3_1Em2_1_08/homogenization.txt",
    delimiter="|", usecols = numpy.arange(1,30), skiprows = 1)

D_250_15Ep3_1Em2_1_10 = \
  numpy.loadtxt(
    "/home/iwtm36/iwtm36/gCP/results/final_report/250_15Ep3_1Em2_1_10/homogenization.txt",
    delimiter="|", usecols = numpy.arange(1,30), skiprows = 1)

x = [0, 2, 4, 6, 10]

y = [D_250_15Ep3_1Em2_1_02[39,1], D_250_15Ep3_1Em2_1_02[-1,1], D_250_15Ep3_1Em2_1_04[-1,1],D_250_15Ep3_1Em2_1_06[-1,1],D_250_15Ep3_1Em2_1_10[-1,1]]

matplotlib.pyplot.plot(x,y)

matplotlib.pyplot.show()


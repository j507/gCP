import math
import numpy as np
import decimal
decimal.getcontext().prec = 28
# 1 - Time
# 2 - Displacement x
# 3 - Displacement y
# 4 - Displacement z
# 5 - Displacement magnitude
# 6 - Equivalent edge dislocation density
# 7 - Equivalent plastic strain
# 8 - Equivalent screw dislocation density
# 9 - Slip 1
# 10 - Slip 2
# 11 - Strain 11
# 12 - Strain 12x2
# 13 - Strain 13x2
# 14 - Strain 22
# 15 - Strain 23x2
# 16 - Strain 33
# 17 - Stress 11
# 18 - Stress 12
# 19 - Stress 13
# 20 - Stress 22
# 21 - Stress 23
# 22 - Stress 33
# 23 - Von Mises plastic strain
# 24 - Von Mises stress

erf_1Em8_1 = np.loadtxt(
  "/calculate/results/rate_dependence/erf_1Em8_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

erf_1Em8_1_long = np.loadtxt(
  "/calculate/results/rate_dependence/erf_1Em8_1_long/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

erf_1Em7_1 = np.loadtxt(
  "/calculate/results/rate_dependence/erf_1Em7_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

erf_1Em7_2 = np.loadtxt(
  "/calculate/results/rate_dependence/erf_1Em7_2/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

erf_1Em7_4 = np.loadtxt(
  "/calculate/results/rate_dependence/erf_1Em7_4/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

erf_1Em6_1 = np.loadtxt(
  "/calculate/results/rate_dependence/erf_1Em6_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

erf_1Em5_1 = np.loadtxt(
  "/calculate/results/rate_dependence/erf_1Em5_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

erf_1Em4_1 = np.loadtxt(
  "/calculate/results/rate_dependence/erf_1Em4_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

erf_1Em3_1 = np.loadtxt(
  "/calculate/results/rate_dependence/erf_1Em3_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

erf = [erf_1Em3_1, erf_1Em4_1, erf_1Em5_1, erf_1Em6_1, erf_1Em7_1, erf_1Em8_1]

erf_max = [np.max(erf[0], axis=0),
           np.max(erf[1], axis=0),
           np.max(erf[2], axis=0),
           np.max(erf[3], axis=0),
           np.max(erf[4], axis=0),
           np.max(erf[5], axis=0),
          ]

tanh_1Em8_1 = np.loadtxt(
  "/calculate/results/rate_dependence/tanh_1Em8_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

tanh_1Em7_1 = np.loadtxt(
  "/calculate/results/rate_dependence/tanh_1Em7_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

tanh_1Em6_1 = np.loadtxt(
  "/calculate/results/rate_dependence/tanh_1Em6_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

tanh_1Em5_1 = np.loadtxt(
  "/calculate/results/rate_dependence/tanh_1Em5_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

tanh_1Em4_1 = np.loadtxt(
  "/calculate/results/rate_dependence/tanh_1Em4_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

tanh_1Em3_1 = np.loadtxt(
  "/calculate/results/rate_dependence/tanh_1Em3_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

tanh = [tanh_1Em3_1, tanh_1Em4_1, tanh_1Em5_1, tanh_1Em6_1, tanh_1Em7_1, tanh_1Em8_1]

tanh_max = [np.max(tanh[0], axis=0),
            np.max(tanh[1], axis=0),
            np.max(tanh[2], axis=0),
            np.max(tanh[3], axis=0),
            np.max(tanh[4], axis=0),
            np.max(tanh[5], axis=0),
           ]

gd_1Em8_1 = np.loadtxt(
  "/calculate/results/rate_dependence/gd_1Em8_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

gd_1Em7_1 = np.loadtxt(
  "/calculate/results/rate_dependence/gd_1Em7_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

gd_1Em6_1 = np.loadtxt(
  "/calculate/results/rate_dependence/gd_1Em6_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

gd_1Em5_1 = np.loadtxt(
  "/calculate/results/rate_dependence/gd_1Em5_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

gd_1Em4_1 = np.loadtxt(
  "/calculate/results/rate_dependence/gd_1Em4_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

gd_1Em3_1 = np.loadtxt(
  "/calculate/results/rate_dependence/gd_1Em3_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

gd = [gd_1Em3_1, gd_1Em4_1, gd_1Em5_1, gd_1Em6_1, gd_1Em7_1, gd_1Em8_1]

gd_max = [np.max(gd[0], axis=0),
          np.max(gd[1], axis=0),
          np.max(gd[2], axis=0),
          np.max(gd[3], axis=0),
          np.max(gd[4], axis=0),
          np.max(gd[5], axis=0),
         ]

sqrt_1Em8_1 = np.loadtxt(
  "/calculate/results/rate_dependence/sqrt_1Em8_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

sqrt_1Em7_1 = np.loadtxt(
  "/calculate/results/rate_dependence/sqrt_1Em7_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

sqrt_1Em6_1 = np.loadtxt(
  "/calculate/results/rate_dependence/sqrt_1Em6_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

sqrt_1Em5_1 = np.loadtxt(
  "/calculate/results/rate_dependence/sqrt_1Em5_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

sqrt_1Em4_1 = np.loadtxt(
  "/calculate/results/rate_dependence/sqrt_1Em4_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

sqrt_1Em3_1 = np.loadtxt(
  "/calculate/results/rate_dependence/sqrt_1Em3_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

sqrt = [sqrt_1Em3_1, sqrt_1Em4_1, sqrt_1Em5_1, sqrt_1Em6_1, sqrt_1Em7_1, sqrt_1Em8_1]

sqrt_max = [np.max(sqrt[0], axis=0),
            np.max(sqrt[1], axis=0),
            np.max(sqrt[2], axis=0),
            np.max(sqrt[3], axis=0),
            np.max(sqrt[4], axis=0),
            np.max(sqrt[5], axis=0),
           ]

atan_1Em8_1 = np.loadtxt(
  "/calculate/results/rate_dependence/atan_1Em8_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

atan_1Em7_1 = np.loadtxt(
  "/calculate/results/rate_dependence/atan_1Em7_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

atan_1Em6_1 = np.loadtxt(
  "/calculate/results/rate_dependence/atan_1Em6_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

atan_1Em5_1 = np.loadtxt(
  "/calculate/results/rate_dependence/atan_1Em5_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

atan_1Em4_1 = np.loadtxt(
  "/calculate/results/rate_dependence/atan_1Em4_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

atan_1Em3_1 = np.loadtxt(
  "/calculate/results/rate_dependence/atan_1Em3_1/plot_selection_over_time.csv",
  delimiter=",", skiprows = 1)

atan = [atan_1Em3_1, atan_1Em4_1, atan_1Em5_1, atan_1Em6_1, atan_1Em7_1, atan_1Em8_1]

atan_max = [np.max(atan[0], axis=0),
            np.max(atan[1], axis=0),
            np.max(atan[2], axis=0),
            np.max(atan[3], axis=0),
            np.max(atan[4], axis=0),
            np.max(atan[5], axis=0),
           ]

erf_softnening = []
tanh_softnening = []
gd_softnening = []
sqrt_softnening = []
atan_softnening = []

erf_accuracy = []
tanh_accuracy = []
gd_accuracy = []
sqrt_accuracy = []
atan_accuracy = []

erf_decay = np.zeros((10,6))
tanh_decay = np.zeros((10,6))
gd_decay = np.zeros((10,6))
sqrt_decay = np.zeros((10,6))
atan_decay = np.zeros((10,6))

erf_rate_dependence = np.zeros((10,3))

erf_decay_long = np.zeros((100, 1))

for i in range(6):
  erf_softnening.append(1.0-erf[i][199, 18] / erf_max[i][18])
  tanh_softnening.append(1.0-tanh[i][199, 18] / tanh_max[i][18])
  gd_softnening.append(1.0-gd[i][199, 18] / gd_max[i][18])
  sqrt_softnening.append(1.0-sqrt[i][199, 18] / sqrt_max[i][18])
  atan_softnening.append(1.0-atan[i][199, 18] / atan_max[i][18])

  erf_accuracy.append(1.0-erf[i][199, 18] / erf_max[5][18])
  tanh_accuracy.append(1.0-tanh[i][199, 18] / erf_max[5][18])
  gd_accuracy.append(1.0-gd[i][199, 18] / erf_max[5][18])
  sqrt_accuracy.append(1.0-sqrt[i][199, 18] / erf_max[5][18])
  atan_accuracy.append(1.0-atan[i][199, 18] / erf_max[5][18])

for i in range(10):
  for j in range(6):
    erf_decay[i,j] = (1.0 - erf[j][399 + i*200, 18] / erf_max[j][18])
    tanh_decay[i,j] = (1.0 - tanh[j][399 + i*200, 18] / tanh_max[j][18])
    gd_decay[i,j] = (1.0 - gd[j][399 + i*200, 18] / gd_max[j][18])
    sqrt_decay[i,j] = (1.0 - sqrt[j][399 + i*200, 18] / sqrt_max[j][18])
    atan_decay[i,j] = (1.0 - atan[j][399 + i*200, 18] / atan_max[j][18])

for i in range(10):
    erf_rate_dependence[i,0] = (1.0 - erf_1Em7_1[399 + i*200, 18] / erf_max[4][18])
    erf_rate_dependence[i,1] = (1.0 - erf_1Em7_2[599 + i*400, 18] / erf_max[4][18])
    erf_rate_dependence[i,2] = (1.0 - erf_1Em7_4[999 + i*800, 18] / erf_max[4][18])

for i in range(100):
    erf_decay_long[i,0] = (1.0 - erf_1Em8_1_long[399 + i*200, 18] / erf_max[5][18])

erf_test = np.transpose(erf_decay)
erf_test   = np.c_[np.arange(1,7), erf_test]

erf_decay   = np.c_[np.arange(1,11), erf_decay]
tanh_decay  = np.c_[np.arange(1,11), tanh_decay]
gd_decay    = np.c_[np.arange(1,11), gd_decay]
sqrt_decay  = np.c_[np.arange(1,11), sqrt_decay]
atan_decay  = np.c_[np.arange(1,11), atan_decay]
erf_rate_dependence = np.c_[np.arange(1,11), erf_rate_dependence]
erf_decay_long = np.c_[np.arange(1,101), erf_decay_long]

np.savetxt('erf_decay.csv', erf_decay, delimiter=',')
np.savetxt('tanh_decay.csv', tanh_decay, delimiter=',')
np.savetxt('gd_decay.csv', gd_decay, delimiter=',')
np.savetxt('sqrt_decay.csv', sqrt_decay, delimiter=',')
np.savetxt('atan_decay.csv', atan_decay, delimiter=',')
np.savetxt('erf_rate_dependence.csv', erf_rate_dependence, delimiter=',')
np.savetxt('erf_decay_long.csv', erf_decay_long, delimiter=',')
np.savetxt('erf_test.csv', erf_test, delimiter=',')


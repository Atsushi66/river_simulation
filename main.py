import numpy as np
import matplotlib.pyplot as plt

#  ITEMS FOR NUMERICAL CALCULATION
tvec = np.arange(0, 10, 0.07)  # time vector
dvec = np.arange(0, 700, 7)  # distance vector
volume_element = 2.734

# MAXIMUM GROWTH RATE OF BACTERIA
u_max = 2.0

# HALF SATURATION CONSTANTS
KS = 2.0

# YIELD COEFFICIENTS
Y = 1/1.9

# DECAY AND MICROBIAL DETACHMENT COEFFICIENTS
b = 0.05

#
cS = np.empty((tvec.size, dvec.size), dtype='float64')
cO2 = np.empty((tvec.size, dvec.size), dtype='float64')
cX = np.empty((tvec.size, dvec.size), dtype='float64')
rS = np.empty((tvec.size, dvec.size), dtype='float64')
rO2 = np.empty((tvec.size, dvec.size), dtype='float64')
rX = np.empty((tvec.size, dvec.size), dtype='float64')
rX_rk1 = np.empty((tvec.size, dvec.size), dtype='float64')
rX_rk2 = np.empty((tvec.size, dvec.size), dtype='float64')
rX_rk3 = np.empty((tvec.size, dvec.size), dtype='float64')
rX_rk4 = np.empty((tvec.size, dvec.size), dtype='float64')



# Initial and boundary conditions
cS[0, :] = 0
cS[:, 0] = 3.31
cO2[0, :] = 0
cO2[:, 0] = 5.0
cX[0, :] = 0
cX[:, 0] = 8.16
rS[0, :] = 0
rS[:, 0] = 0
rO2[0, :] = 0
rO2[:, 0] = 0
rX[0, :] = 0
rX[:, 0] = 0
qflow = 8031.85

# turning point of seasonal variation
dt = 0.005  # time step


# Main loop
for k in range(1, tvec.size-1):
    for l in range(1, dvec.size):

        # (1) concentration of bacteria
        rX_rk1[k, l] = u_max * cS[k, l] /(KS + cS[k, l]) *cX[k, l] - b * cX[k, l]
        diff1_rk1 = (qflow / volume_element *(cX[k, l-1] - cX[k, l]) + rX_rk1[k, l]) * dt

        rX_rk2[k, l] = u_max * (cS[k, l] + 1/2 *diff1_rk1) /(KS + (cS[k, l] + 1/2 *diff1_rk1)) *(cX[k, l] + 1/2 *diff1_rk1) - b * (cX[k, l] + 1/2 *diff1_rk1)
        diff1_rk2 = (qflow / volume_element *(cX[k, l-1] - (cX[k, l] + 1/2 *diff1_rk1)) + rX_rk2[k, l]) * dt

        rX_rk3[k, l] = u_max * (cS[k, l] + 1/2 *diff1_rk2) /(KS + (cS[k, l] + 1/2 *diff1_rk2)) *(cX[k, l] + 1/2 *diff1_rk2) - b * (cX[k, l] + 1/2 *diff1_rk2)
        diff1_rk3 = (qflow / volume_element *(cX[k, l-1] - (cX[k, l] + 1/2 *diff1_rk2)) + rX_rk1[k, l]) * dt

        rX_rk4[k, l] = u_max * (cS[k, l] + diff1_rk3) /(KS + (cS[k, l] + diff1_rk3)) *(cX[k, l] + diff1_rk3) - b * (cX[k, l] + diff1_rk3)
        diff1_rk4 = (qflow / volume_element *(cX[k, l-1] - (cX[k, l] + diff1_rk3)) + rX_rk1[k, l]) * dt

        diff1_rk = 1/6 *(diff1_rk1 + 2 * diff1_rk2 + 2 * diff1_rk3 + diff1_rk4)
    
        cX[k+1, l] = cX[k, l] + diff1_rk

        if cX[k+1, l] < 0:
            cX[k+1, l] = 0

        # (2) concentration of BOD
        rS[k, l] = - 1 / Y * u_max * cS[k, l] /(KS + cS[k, l]) * cX[k, l]
        diff2_rk1 = (qflow / volume_element *(cX[k, l-1] - cX[k, l]) + rS_rk1[k, l]) * dt

        

        cS[k+1, l] = cS[k, l] + diff2

        if cS[k+1, l] < 0:
            cS[k+1, l] = 0

        # (3) concentration of DO
        rO2[k, l] = - 1 / Y * u_max * cS[k, l] /(KS + cS[k, l]) *cX[k, l]

        diff3 = (qflow / volume_element *(cX[k, l-1] - cX[k, l]) + rO2[k, l]) * dt

        cO2[k+1, l] = cO2[k, l] + diff3

        if cO2[k+1, l] < 0:
            cO2[k+1, l] = 0




# plot results
plt.subplot(3, 1, 1)
plt.plot(tvec, cX[:, -1])
plt.xlabel('time[day]')
plt.ylabel('SS concentration')
plt.ylim(0, cX.max())
plt.subplot(3, 1, 2)
plt.plot(tvec, cS[:, -1])
plt.xlabel('time[day]')
plt.ylabel('BOD concentraion')
plt.ylim(0, cS.max())
plt.subplot(3, 1, 3)
plt.plot(tvec, cO2[:, -1])
plt.xlabel('time[day]')
plt.ylabel('DO concentration')
plt.ylim(0, cO2.max())
plt.show()



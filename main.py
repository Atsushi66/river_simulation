import numpy as np
from numpy import log10
import matplotlib.pyplot as plt
from collections import namedtuple
from IPython.core.pylabtools import figsize
from util import runge_kutta

# 排水流入ポイント
a = 334
b = 531

#  ITEMS FOR NUMERICAL CALCULATION
tvec = np.arange(0, 0.1, 0.001)  # time vector
dvec = np.arange(0, 7775.04, 7.12)  # distance vector
v1 = 2.734
v2 = 3.752
v3 = 4.728

cSS = np.empty((tvec.size, dvec.size), dtype='float64')
cSI = np.empty((tvec.size, dvec.size), dtype='float64')
cSNH4 = np.empty((tvec.size, dvec.size), dtype='float64')
cSNH3 = np.empty((tvec.size, dvec.size), dtype='float64')
cSNO2 = np.empty((tvec.size, dvec.size), dtype='float64')
cSNO3 = np.empty((tvec.size, dvec.size), dtype='float64')
cSHPO4 = np.empty((tvec.size, dvec.size), dtype='float64')
cSH2PO4 = np.empty((tvec.size, dvec.size), dtype='float64')
cSO2 = np.empty((tvec.size, dvec.size), dtype='float64')
cSCO2 = np.empty((tvec.size, dvec.size), dtype='float64')
cSHCO3 = np.empty((tvec.size, dvec.size), dtype='float64')
cSCO3 = np.empty((tvec.size, dvec.size), dtype='float64')
cSH = np.empty((tvec.size, dvec.size), dtype='float64')
cSOH = np.empty((tvec.size, dvec.size), dtype='float64')
cSCa = np.empty((tvec.size, dvec.size), dtype='float64')
cXH = np.empty((tvec.size, dvec.size), dtype='float64')
cXN1 = np.empty((tvec.size, dvec.size), dtype='float64')
cXN2 = np.empty((tvec.size, dvec.size), dtype='float64')
cXALG = np.empty((tvec.size, dvec.size), dtype='float64')
cXCON = np.empty((tvec.size, dvec.size), dtype='float64')
cXS = np.empty((tvec.size, dvec.size), dtype='float64')
cXI = np.empty((tvec.size, dvec.size), dtype='float64')

# Initial and boundary conditions
cSS[0, :] = 0
cSS[:, 0] = 3.31
cSI[0, :] = 0
cSI[:, 0] = 0
cSNH4[0, :] = 0 
cSNH4[:, 0] = 0
cSNH3[0, :] = 0
cSNH3[:, 0] = 0
cSNO2[0, :] = 0
cSNO2[:, 0] = 0
cSNO3[0, :] = 0
cSNO3[:, 0] = 0
cSHPO4[0, :] = 0
cSHPO4[:, 0] = 0
cSH2PO4[0, :] = 0
cSH2PO4[:, 0] = 0
cSO2[0, :] = 0
cSO2[:, 0] = 0
cSCO2[0, :] = 0
cSCO2[:, 0] = 0
cSHCO3[0, :] = 0
cSHCO3[:, 0] = 0
cSCO3[0, :] = 0
cSCO3[:, 0] = 0
cSH[0, :] = 0
cSH[:, 0] = 0
cSOH[0, :] = 0
cSOH[:, 0] = 0
cSCa[0, :] = 0
cSCa[:, 0] = 0
cXH[0, :] = 0
cXH[:, 0] = 0
cXN1[0, :] = 0
cXN1[:, 0] = 0
cXN2[0, :] = 0
cXN2[:, 0] = 0
cXALG[0, :] = 0
cXALG[:, 0] = 0
cXCON[0, :] = 0
cXCON[:, 0] = 0
cXS[0, :] = 0
cXS[:, 0] = 0
cXI[0, :] = 0
cXI[:, 0] = 0

# 初期値
SO2in0=5.0
SCO2in0=0.1
SHCO3in0=0.1
SCO3in0=0.1
SHin0=0.0001
SOHin0=0.0001
SCain0=0.1
XHin0=10.0
XN1in0=0.1
XN2in0=0.05
XALGin0=1.0
XCONin0=0.1

# turning point of seasonal variation
dt = 0.001  # time step

# Main loop
for i in range(1, tvec.size-1):
    for j in range(1, dvec.size):
        # i is time, j is distance
        
        # time evolution
        FlowConst = namedtuple('FlowConst',
                                ['q1', 'q2', 'q3',
                                 'u1', 'u2', 'u3',
                                 'T','Ie',
                                 'SSin0a','SSin0b','SSin0c',
                                 'SIin0a','SIin0b','SIin0c',
                                 'XSin0a','XSin0b','XSin0c',
                                 'XIin0a','XIin0b','XIin0c',
                                 'SNH4in0a','SNH4in0b','SNH4in0c',
                                 'SNH3in0a','SNH3in0b','SNH3in0c',
                                 'SNO2in0a','SNO2in0b','SNO2in0c',
                                 'SNO3in0a','SNO3in0b','SNO3in0c',
                                 'SHPO4in0a','SHPO4in0b','SHPO4in0c',
                                 'SH2PO4in0a','SH2PO4in0b','SH2PO4in0c',
                                 'SNO2inGa','SNO2inGb','SNO2inGc',
                                 'SNO3inGa','SNO3inGb','SNO3inGc',
                                 'Ma1','Ma2','Ma3'
                                ])
        if i*dt<=0.7:
            fc = FlowConst(8031.85, 13205.99, 19060.29, 
                            19161.05, 23119.85, 26497.44, 
                            13.30931, 184.8061574, 
                            3.31, 1.75, 1.27, 
                            15.76, 8.52, 6.29, 
                            8.16, 4.16, 3.05, 
                            8.84, 4.58, 3.41, 
                            12.23, 6.23, 4.63, 
                            1.36, 0.69, 0.51,
                            0.44, 0.25,0.18,
                            0.92, 0.41,0.30,
                            0.64, 0.34,0.25,
                            0.07, 0.04,0.03,
                            0.0037, 0.0022,	0.0015,	
                            0.0183,	0.0111,	0.0077,	
                            0.00, 0.00,	0.00)
        
        elif i*dt>0.7 and i*dt<=1.0:
            fc = FlowConst(12999.94, 22171.82, 32738.43, 	
                            22984.08, 28010.64, 32254.68,
                            17.29397, 166.9637407,	
                            2.05, 1.04,	0.74,
                            9.73, 5.08,	3.66,
                            5.04, 2.48,	1.78,
                            5.46, 2.73,	1.99, 
                            7.55, 3.71,	2.69, 
                            0.84, 0.41,	0.30, 
                            0.27, 0.15,	0.11, 
                            0.57, 0.25,	0.18, 
                            0.40, 0.20,	0.15, 
                            0.04, 0.02,	0.02, 
                            0.0023, 0.0013, 0.0009,
                            0.0113,	0.0066,	0.0045, 
                            0.00, 0.0017, 0.0017)
                
        elif i*dt>1.0 and i*dt<=1.3:
            fc = FlowConst(15141.35,26036.41,38634.18,
                            24331.20, 29699.33, 34212.80,
                            20.17178, 155.836575,
                            1.76, 0.89, 0.63,
                            8.36, 4.32, 3.10,
                            4.33, 2.11, 1.50,
                            4.69, 2.32, 1.68,
                            6.49, 3.16, 2.28,
                            0.72, 0.35, 0.25,
                            0.23, 0.13, 0.09,
                            0.49, 0.21, 0.15,
                            0.34, 0.17, 0.13,
                            0.04, 0.02, 0.01,
                            0.0019, 0.0011, 0.0008,
                            0.0097, 0.0057, 0.0038,
                            0.00, 0.0017, 0.0017)

        elif i*dt>1.3 and i*dt<=1.6:
            fc = FlowConst(22679.14,39639.74,59387.22,
                            28243.33,34525.09,39740.23,
                            22.31169,139.893237,
                            1.17,0.58,0.41,
                            5.58,2.84,2.02,
                            2.89,1.39,0.98,
                            3.13,1.52,1.1,
                            4.33,2.08,1.48,
                            0.48,0.23,0.16,
                            0.16,0.08,0.06,
                            0.33,0.14,0.1,
                            0.23,0.11,0.08,
                            0.03,0.01,0.01,
                            0.0013,0.0007,0.0005,
                            0.0065,0.0037,0.0025,
                            0.0017,0.0017,0)

            
        elif i*dt>1.6 and i*dt<=1.9:
            fc = FlowConst(12400.34,21089.74,31087.62,
                            22580.69,27501.86,31662.03,
                            23.86128,169.6089207,
                            2.14,1.1,0.78,
                            10.2,5.34,3.86,
                            5.28,2.61,1.87,
                            5.73,2.87,2.09,
                            7.92,3.9,2.84,
                            0.88,0.43,0.32,
                            0.28,0.16,0.11,
                            0.6,0.26,0.19,
                            0.42,0.21,0.16,
                            0.05,0.02,0.02,
                            0.0024,0.0014,0.0009,
                            0.0119,0.007,0.0047,
                            0,0.0017,0.0017)

            
        elif i*dt>1.9 and i*dt<=2.2:
            fc = FlowConst(18567.62,32219.74,48067.38,
                            26242.73,32070.78,36941.03,
                            20.39315,139.0493691,
                            1.43,0.72,0.51,
                            6.82,3.49,2.49,
                            3.53,1.71,1.21,
                            3.83,1.88,1.35,
                            5.29,2.55,1.83,
                            0.59,0.28,0.2,
                            0.19,0.1,0.07,
                            0.4,0.17,0.12,
                            0.28,0.14,0.1,
                            0.03,0.02,0.01,
                            0.0016,0.0009,0.0006,
                            0.0079,0.0046,0.0031,
                            0,0.0017,0.0017)
            
        elif i*dt>2.2 and i*dt<=2.5:
            fc = FlowConst(24477.93,42885.99,64339.65,
                            29040.62,35495.99,40841.11,
                            17.44155,125.4075729,
                            1.09,0.54,0.38,
                            5.17,2.62,1.86,
                            2.68,1.28,0.9,
                            2.9,1.41,1.01,
                            4.01,1.92,1.37,
                            0.45,0.21,0.15,
                            0.14,0.08,0.05,
                            0.3,0.13,0.09,
                            0.21,0.11,0.08,
                            0.02,0.01,0.01,
                            0.0012,0.0007,0.0005,
                            0.006,0.0034,0.0023,
                            0.0017,0.0017,0.0017)

        elif i*dt>2.5 and i*dt<=2.8:
            fc = FlowConst(11115.49,18770.99,27550.17,
                            21670.69,26348.42,30313.47,
                            13.01415,116.5247729,
                            2.39,1.23,0.88,
                            11.38,5.99,4.35,
                            5.89,2.93,2.11,
                            6.39,3.22,2.36,
                            8.84,4.38,3.2,
                            0.98,0.49,0.36,
                            0.32,0.18,0.13,
                            0.66,0.29,0.21,
                            0.46,0.24,0.18,
                            0.05,0.03,0.02,
                            0.0026,0.0016,0.0011,
                            0.0132,0.0078,0.0053,
                            0,0,0.0017)

        elif i*dt>2.8 and i*dt<=3.1:
            fc = FlowConst(15912.26,27427.66,40756.65,
                            24784.86,30264.65,34865.38,
                            9.17707,99.97320602,
                            1.67,0.84,0.6,
                            7.95,4.1,2.94,
                            4.12,2,1.43,
                            4.46,2.2,1.6,
                            6.17,3,2.16,
                            0.69,0.33,0.24,
                            0.22,0.12,0.09,
                            0.46,0.2,0.14,
                            0.32,0.16,0.12,
                            0.04,0.02,0.01,
                            0.0018,0.0011,0.0007,
                            0.0092,0.0054,0.0036,
                            0,0.0017,0.0017)

            
        elif i*dt>3.1 and i*dt<=3.4:
           fc = FlowConst(6061.75,9650.57,13636.2,
                            17207.22,20544.32,23398.86,
                            7.70127,120.1430125,
                            4.39,2.4,1.78,
                            20.88,11.66,8.79,
                            10.81,5.69,4.26,
                            11.72,6.26,4.77,
                            16.2,8.53,6.46,
                            1.8,0.95,0.72,
                            0.58,0.34,0.26,
                            1.22,0.56,0.42,
                            0.85,0.47,0.35,
                            0.09,0.05,0.04,
                            0.0048,0.003,0.0022,
                            0.0243,0.0152,0.0108,
                            0,0,0)

            
        elif i*dt>3.4 and i*dt<=3.7:
            fc = FlowConst(4262.96,6404.32,8683.77,
                            15025.49,17573.67,19738.19,
                            8.88191,177.2605688,
                            6.24,3.61,2.8,
                            29.68,17.57,13.81,
                            15.37,8.58,6.7,
                            16.66,9.44,7.49,
                            23.04,12.85,10.15,
                            2.56,1.43,1.13,
                            0.83,0.52,0.41,
                            1.73,0.85,0.66,
                            1.21,0.7,0.56,
                            0.13,0.08,0.06,
                            0.0069,0.0046,0.0034,
                            0.0345,0.023,0.0169,
                            0,0,0)


            
        elif i*dt>3.7 and i*dt<=4.0:
            fc = FlowConst(5547.81,8723.07,12221.22,
                            16631.69,19772.07,22457.69,
                            10.57908,194.4688115,	
                            4.79,2.65,1.99,
                            22.81,12.9,9.81,
                            11.81,6.3,4.76,
                            12.81,6.93,5.32,
                            17.7,9.44,7.21,
                            1.97,1.05,0.8,
                            0.64,0.38,0.29,
                            1.33,0.63,0.47,
                            0.93,0.52,0.4,
                            0.1,0.06,0.04,
                            0.0053,0.0034,0.0024,
                            0.0265,0.0169,0.012,
                            0.00,0.00,0.00)
	
        # space evolution
        if j == 1:
            SSin=fc.SSin0a 
            SIin=fc.SIin0a 
            SNH4in=fc.SNH4in0a 
            SNH3in=fc.SNH3in0a  
            SNO2in=fc.SNO2in0a+fc.SNO2inGa
            SNO3in=fc.SNO3in0a+fc.SNO3inGa
            SHPO4in=fc.SHPO4in0a
            SH2PO4in=fc.SH2PO4in0a
            SO2in=SO2in0
            SCO2in=SCO2in0
            SHCO3in=SHCO3in0
            SCO3in=SCO3in0
            SHin=SHin0
            SOHin=SOHin0
            SCain=SCain0
            XHin=XHin0
            XN1in=XN1in0
            XN2in=XN2in0
            XALGin=XALGin0
            XCONin=XCONin0
            XSin=fc.XSin0a
            XIin=fc.XIin0a
            # TSin=fc.TSin0
            q=fc.q1
            vb=fc.u1
            v=v1
            Ma=fc.Ma1
        
        
        elif j>=2 and j<=a-1:
            SSin=cSS[i, j-1]
            SIin=cSI[i, j-1]
            SNH4in=cSNH4[i, j-1]
            SNH3in=cSNH3[i, j-1]
            SNO2in=cSNO2[i, j-1]+fc.SNO2inGa
            SNO3in=cSNO3[i, j-1]+fc.SNO3inGa
            SHPO4in=cSHPO4[i, j-1]
            SH2PO4in=cSH2PO4[i, j-1]
            SO2in=cSO2[i, j-1]
            SCO2in=cSCO2[i, j-1]
            SHCO3in=cSHCO3[i, j-1]
            SCO3in=cSCO3[i, j-1]
            SHin=cSH[i, j-1]
            SOHin=cSOH[i, j-1]
            SCain=cSCa[i, j-1]
            XHin=cXH[i, j-1]
            XN1in=cXN1[i, j-1]
            XN2in=cXN2[i, j-1]
            XALGin=cXALG[i, j-1]
            XCONin=cXCON[i, j-1]
            XSin=cXS[i, j-1]
            XIin=cXI[i, j-1]
            #TSin=cTS[i, j-1]
            q=fc.q1
            vb=fc.u1
            v=v1
            Ma=fc.Ma1
        
        elif j==a:
            SSin=cSS[i, j-1]+fc.SSin0b
            SIin=cSI[i, j-1]+fc.SIin0b
            SNH4in=cSNH4[i, j-1]+fc.SNH4in0b
            SNH3in=cSNH3[i, j-1]+fc.SNH3in0b
            SNO2in=cSNO2[i, j-1]+fc.SNO2in0b+fc.SNO2inGb
            SNO3in=cSNO3[i, j-1]+fc.SNO3in0b+fc.SNO3inGb
            SHPO4in=cSHPO4[i, j-1]+fc.SHPO4in0b
            SH2PO4in=cSH2PO4[i, j-1]+fc.SH2PO4in0b
            SO2in=cSO2[i, j-1]
            SCO2in=cSCO2[i, j-1]
            SHCO3in=cSHCO3[i, j-1]
            SCO3in=cSCO3[i, j-1]
            SHin=cSH[i, j-1]
            SOHin=cSOH[i, j-1]
            SCain=cSCa[i, j-1]
            XHin=cXH[i, j-1]
            XN1in=cXN1[i, j-1]
            XN2in=cXN2[i, j-1]
            XALGin=cXALG[i, j-1]
            XCONin=cXCON[i, j-1]
            XSin=cXS[i, j-1]+fc.XSin0b
            XIin=cXI[i, j-1]+fc.XIin0b
            #TSin=cTS[i, j-1]
            q=fc.q2
            vb=fc.u2
            v=v2
            Ma=fc.Ma2
        
        elif j>=a+1 and j<=b-1:
            SSin=cSS[i, j-1]
            SIin=cSI[i, j-1]
            SNH4in=cSNH4[i, j-1]
            SNH3in=cSNH3[i, j-1]
            SNO2in=cSNO2[i, j-1]+fc.SNO2inGb
            SNO3in=cSNO3[i, j-1]+fc.SNO3inGb
            SHPO4in=cSHPO4[i, j-1]
            SH2PO4in=cSH2PO4[i, j-1]
            SO2in=cSO2[i, j-1]
            SCO2in=cSCO2[i, j-1]
            SHCO3in=cSHCO3[i, j-1]
            SCO3in=cSCO3[i, j-1]
            SHin=cSH[i, j-1]
            SOHin=cSOH[i, j-1]
            SCain=cSCa[i, j-1]
            XHin=cXH[i, j-1]
            XN1in=cXN1[i, j-1]
            XN2in=cXN2[i, j-1]
            XALGin=cXALG[i, j-1]
            XCONin=cXCON[i, j-1]
            XSin=cXS[i, j-1]
            XIin=cXI[i, j-1]
            # TSin=cTS[i, j-1]
            q=fc.q2
            vb=fc.u2
            v=v2
            Ma=fc.Ma2
        
        elif j == b:
            SSin=cSS[i, j-1]+fc.SSin0c
            SIin=cSI[i, j-1]+fc.SIin0c
            SNH4in=cSNH4[i, j-1]+fc.SNH4in0c
            SNH3in=cSNH3[i, j-1]+fc.SNH3in0c
            SNO2in=cSNO2[i, j-1]+fc.SNO2in0c+fc.SNO2inGc
            SNO3in=cSNO3[i, j-1]+fc.SNO3in0c+fc.SNO3inGc
            SHPO4in=cSHPO4[i, j-1]+fc.SHPO4in0c
            SH2PO4in=cSH2PO4[i, j-1]+fc.SH2PO4in0c
            SO2in=cSO2[i, j-1]
            SCO2in=cSCO2[i, j-1]
            SHCO3in=cSHCO3[i, j-1]
            SCO3in=cSCO3[i, j-1]
            SHin=cSH[i, j-1]
            SOHin=cSOH[i, j-1]
            SCain=cSCa[i, j-1]
            XHin=cXH[i, j-1]
            XN1in=cXN1[i, j-1]
            XN2in=cXN2[i, j-1]
            XALGin=cXALG[i, j-1]
            XCONin=cXCON[i, j-1]
            XSin=cXS[i, j-1]+fc.XSin0c
            XIin=cXI[i, j-1]+fc.XIin0c
            # TSin=cTS[i, j-1]
            q=fc.q3
            vb=fc.u3
            v=v3
            Ma=fc.Ma3
        
        elif j>=b+1:
            SSin=cSS[i, j-1]
            SIin=cSI[i, j-1]
            SNH4in=cSNH4[i, j-1]
            SNH3in=cSNH3[i, j-1]
            SNO2in=cSNO2[i, j-1]+fc.SNO2inGc
            SNO3in=cSNO3[i, j-1]+fc.SNO3inGc
            SHPO4in=cSHPO4[i, j-1]
            SH2PO4in=cSH2PO4[i, j-1]
            SO2in=cSO2[i, j-1]
            SCO2in=cSCO2[i, j-1]
            SHCO3in=cSHCO3[i, j-1]
            SCO3in=cSCO3[i, j-1]
            SHin=cSH[i, j-1]
            SOHin=cSOH[i, j-1]
            SCain=cSCa[i, j-1]
            XHin=cXH[i, j-1]
            XN1in=cXN1[i, j-1]
            XN2in=cXN2[i, j-1]
            XALGin=cXALG[i, j-1]
            XCONin=cXCON[i, j-1]
            XSin=cXS[i, j-1]
            XIin=cXI[i, j-1]
            # TSin=cTS[i, j-1]
            q=fc.q3
            vb=fc.u3
            v=v3
            Ma=fc.Ma3
        
        diff_list = runge_kutta(SSin, SIin, SNH4in, SNH3in, SNO2in, SNO3in, SHPO4in, SH2PO4in, SO2in, SCO2in, SHCO3in, SCO3in, SHin, SOHin, SCain,\
             XHin, XN1in, XN2in, XALGin, XCONin, XSin, XIin, \
                 cSS[i, j], cSI[i, j], cSNH4[i, j], cSNH3[i, j], cSNO2[i, j], cSNO3[i, j], cSHPO4[i, j], cSH2PO4[i, j], cSO2[i, j], cSCO2[i, j], cSHCO3[i, j], cSCO3[i, j], cSH[i, j], cSOH[i, j], cSCa[i, j], \
                     cXH[i, j], cXN1[i, j], cXN2[i, j], cXALG[i, j], cXCON[i, j], cXS[i, j], cXI[i, j], fc.T, q, v, fc.Ie, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, fc.XSin0a, fc.XSin0b, fc.XSin0c, fc.XIin0a, fc.XIin0b, fc.XIin0c)

        # (1) concentration of SS
        cSS[i+1, j] = cSS[i, j] + diff_list[0]

        if cSS[i+1, j] < 0:
            cSS[i+1, j] = 0

        # (2) concentration of SI        
        cSI[i+1, j] = cSI[i, j] + diff_list[1]

        if cSI[i+1, j] < 0:
            cSI[i+1, j] = 0

        # (3) concentration of SNH4       
        cSNH4[i+1, j] = cSNH4[i, j] + diff_list[2]

        if cSNH4[i+1, j] < 0:
            cSNH4[i+1, j] = 0
        
        # (4) concentration of SNH3       
        cSNH3[i+1, j] = cSNH3[i, j] + diff_list[3]

        if cSNH3[i+1, j] < 0:
            cSNH3[i+1, j] = 0
        
        # (5) concentration of SNO2       
        cSNO2[i+1, j] = cSNO2[i, j] + diff_list[4]

        if cSNO2[i+1, j] < 0:
            cSNO2[i+1, j] = 0

        # (6) concentration of SNO3       
        cSNO3[i+1, j] = cSNO3[i, j] + diff_list[5]

        if cSNO3[i+1, j] < 0:
            cSNO3[i+1, j] = 0

        # (7) concentration of SHPO4       
        cSHPO4[i+1, j] = cSHPO4[i, j] + diff_list[6]

        if cSHPO4[i+1, j] < 0:
            cSHPO4[i+1, j] = 0

        # (8) concentration of SH2PO4       
        cSH2PO4[i+1, j] = cSH2PO4[i, j] + diff_list[7]

        if cSH2PO4[i+1, j] < 0:
            cSH2PO4[i+1, j] = 0

        # (9) concentration of SO2       
        cSO2[i+1, j] = cSO2[i, j] + diff_list[8]

        if cSO2[i+1, j] < 0:
            cSO2[i+1, j] = 0

        # (10) concentration of SCO2       
        cSCO2[i+1, j] = cSCO2[i, j] + diff_list[9]

        if cSCO2[i+1, j] < 0:
            cSCO2[i+1, j] = 0

        # (11) concentration of SHCO3       
        cSHCO3[i+1, j] = cSHCO3[i, j] + diff_list[10]

        if cSHCO3[i+1, j] < 0:
            cSHCO3[i+1, j] = 0

        # (12) concentration of SCO3       
        cSCO3[i+1, j] = cSCO3[i, j] + diff_list[11]

        if cSCO3[i+1, j] < 0:
            cSCO3[i+1, j] = 0

        # (13) concentration of SH       
        cSH[i+1, j] = cSH[i, j] + diff_list[12]

        if cSH[i+1, j] < 0:
            cSH[i+1, j] = 0

        # (14) concentration of SOH       
        cSOH[i+1, j] = cSOH[i, j] + diff_list[13]

        if cSOH[i+1, j] < 0:
            cSOH[i+1, j] = 0

        # (15) concentration of SCa       
        cSCa[i+1, j] = cSCa[i, j] + diff_list[14]

        if cSCa[i+1, j] < 0:
            cSCa[i+1, j] = 0

        # (16) concentration of XH       
        cXH[i+1, j] = cXH[i, j] + diff_list[15]

        if cXH[i+1, j] < 0:
            cXH[i+1, j] = 0

        # (17) concentration of XN1      
        cXN1[i+1, j] = cXN1[i, j] + diff_list[16]

        if cXN1[i+1, j] < 0:
            cXN1[i+1, j] = 0
        
        # (18) concentration of XN2       
        cXN2[i+1, j] = cXN2[i, j] + diff_list[17]

        if cXN2[i+1, j] < 0:
            cXN2[i+1, j] = 0

        # (19) concentration of XALG
        cXALG[i+1, j] = cXALG[i, j] + diff_list[18]

        if cXALG[i+1, j] < 0:
            cXALG[i+1, j] = 0

        # (20) concentration of XCON
        cXCON[i+1, j] = cXCON[i, j] + diff_list[19]

        if cXCON[i+1, j] < 0:
            cXCON[i+1, j] = 0

        # (21) concentration of XS
        cXS[i+1, j] = cXS[i, j] + diff_list[20]

        if cXS[i+1, j] < 0:
            cXS[i+1, j] = 0

        # (22) concentration of XI
        cXI[i+1, j] = cXI[i, j] + diff_list[21]

        if cXI[i+1, j] < 0:
            cXI[i+1, j] = 0


# plot results
figsize(12, 18)

plt.subplot(3, 2, 1)
plt.plot(tvec, cSO2[:, a], label='st1')
plt.plot(tvec, cSO2[:, b], label='st2')
plt.xlabel('time')
plt.ylabel('DO concentration')
plt.ylim(0, cSO2.max())

plt.subplot(3, 2, 2)
cBOD = cSS + cXS
plt.plot(tvec, cBOD[:, a], label='st1')
plt.plot(tvec, cBOD[:, b], label='st2')
plt.xlabel('time')
plt.ylabel('BOD concentraion')
plt.ylim(0, cBOD.max())
plt.legend(loc="upper right")

plt.subplot(3, 2, 3)
cXX = cXS + cXN1 + cXN2 + cXALG + cXCON + cXH + cXI
plt.plot(tvec, cXX[:, a], label='st1')
plt.plot(tvec, cXX[:, b], label='st2')
plt.xlabel('time')
plt.ylabel('SS concentration')
plt.ylim(0, cXX.max())
plt.legend(loc="upper right")

plt.subplot(3, 2, 4)
plt.plot(tvec, cSNH4[:, a], label='st1')
plt.plot(tvec, cSNH4[:, b], label='st2')
plt.xlabel('time')
plt.ylabel('SNH4 concentration')
plt.ylim(0, cSNH4.max())
plt.legend(loc="upper right")

plt.subplot(3, 2, 5)
cSNOX = cSNO2 + cSNO3
plt.plot(tvec, cSNOX[:, a], label='st1')
plt.plot(tvec, cSNOX[:, b], label='st2')
plt.xlabel('time')
plt.ylabel('NO2-N + NO3-N concentration')
plt.ylim(0, cSNOX.max())
plt.legend(loc="upper right")

# plt.subplot(3, 2, 6)
# cPh = -log10(10^-3 * cSH)
# plt.plot(tvec, cPh[:, a], label='st1')
# plt.plot(tvec, cPh[:, b], label='st2')
# plt.xlabel('time')
# plt.ylabel('ph concentration')
# plt.ylim(0, cPh.max())
# plt.legend(loc="upper right")


plt.show()



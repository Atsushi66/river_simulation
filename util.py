import numpy as np

w = 9.849
As = 28.48

K2=5.0
Keqw=0.0000000060
Keq1=0.00040
Keq2=0.000000040
KeqN=0.00000034
KeqP=0.000061
Keqso=1.97

dt = 0.001


# IWA RWQM No.1 matrix
def f_SS(SSin, SS, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, XH, XS, T, q, v):
    A = 2.0*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*SO2/(0.2+SO2)*(SNH4+SNH3)/(0.2+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    B = 2.0*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*SO2/(0.2+SO2)*0.2/(0.2+SNH4+SNH3)*SNO3/(0.2+SNO3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    D = 1.6*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*0.2/(0.2+SO2)*SNO3/(0.5+SNO3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    E = 1.6*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*0.2/(0.2+SO2)*SNO2/(0.2+SNO2)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    V = 0.3*np.exp(0.07*(T-20.0))*XS
    return q/v*(SSin - SS) + (-1.9*A -1.9*B -2.2*D -3.7*E +1.0*V)

def f_SI(SIin, SI, q, v):
    return q/v*(SIin - SI)

def f_SNH4(SNH4in, SS, SI, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, SH, SOH, SCa, XH, XN1, XN2, XALG, XCON, XS,  T, q, v, Ie):
    A = 2.0*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*SO2/(0.2+SO2)*(SNH4+SNH3)/(0.2+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    C = 0.2*np.exp(0.07*(T-20.0))*SO2/(0.2+SO2)*XH
    F = 0.1*np.exp(0.07*(T-20.0))*0.2/(0.2+SO2)*SNO3/(0.5+SNO3)*XH
    G = 0.8*np.exp(0.098*(T-20.0))*SO2/(0.5+SO2)*(SNH4+SNH3)/(0.5+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XN1
    H = 0.05*np.exp(0.098*(T-20.0))*SO2/(0.5+SO2)*XN1
    J = 0.05*np.exp(0.069*(T-20.0))*SO2/(0.5+SO2)*XN2
    K = 2.0*np.exp(0.046*(T-20.0))*(SNH4+SNH3+SNO3)/(0.1+SNH4+SNH3+SNO3)*(SNH4+SNH3)/(0.1+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*Ie/500.0*np.exp(1.0-Ie/500.0)*XALG
    M = 0.1*np.exp(0.046*(T-20.0))*SO2/(0.2+SO2)*XALG
    N = 0.1*np.exp(0.046*(T-20.0))*XALG
    O = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XALG*XCON
    P = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XS*XCON
    Q = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XH*XCON
    R = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN1*XCON
    S = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN2*XCON
    TT = 0.05*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XCON
    U = 0.05*np.exp(0.08*(T-20.0))*XCON
    V = 0.3*np.exp(0.07*(T-20.0))*XS
    Z = 0.0001*(SNH4-SH*SNH3/KeqN)
    return q/v*(SNH4in-SNH4)+(-0.012*A +0.071*C +0.071*F -4.8*G +0.071*H +0.071*J -0.065*K +0.058*M +0.029*N +0.13*O +0.13*P +0.45*Q +0.45*R +0.45*S +0.058*TT +0.029*U +0.0*V -1.0*Z)

def f_SNH3(SNH3in, SNH4, SNH3, SH, T, q, v,):
    Z = 0.0001*(SNH4-SH*SNH3/KeqN)
    return q/v*(SNH3in-SNH3)+(1.0*Z)

def f_SNO2(SNO2in, SS, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, XH, XN1, XN2, T, q, v):
    D = 1.6*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*0.2/(0.2+SO2)*SNO3/(0.5+SNO3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    E = 1.6*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*0.2/(0.2+SO2)*SNO2/(0.2+SNO2)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    G = 0.8*np.exp(0.098*(T-20.0))*SO2/(0.5+SO2)*(SNH4+SNH3)/(0.5+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XN1
    I = 1.1*np.exp(0.069*(T-20.0))*SO2/(0.5+SO2)*SNO2/(0.5+SNO2)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XN2
    return q/v*(SNO2in-SNO2)+(1.1*D-1.6*E+4.7*G-21.0*I)

def f_SNO3(SNO3in, SS, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, XH, XN2, XALG, T, q, v, Ie):
    B = 2.0*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*SO2/(0.2+SO2)*0.2/(0.2+SNH4+SNH3)*SNO3/(0.2+SNO3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    D = 1.6*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*0.2/(0.2+SO2)*SNO3/(0.5+SNO3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    F = 0.1*np.exp(0.07*(T-20.0))*0.2/(0.2+SO2)*SNO3/(0.5+SNO3)*XH
    I = 1.1*np.exp(0.069*(T-20.0))*SO2/(0.5+SO2)*SNO2/(0.5+SNO2)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XN2
    LL = 2.0*np.exp(0.046*(T-20.0))*(SNH4+SNH3+SNO3)/(0.1+SNH4+SNH3+SNO3)*0.1/(0.1+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*Ie/500.0*np.exp(1.0-Ie/500.0)*XALG
    return q/v*(SNO3in-SNO3)+(-0.012*B-1.1*D-0.27*F+21.0*I-0.065*LL)

def f_SHPO4(SHPO4in, SS, SI, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, SH, SOH, SCa, XH, XN1, XN2, XALG, XCON, XS,  T, q, v, Ie):
    A = 2.0*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*SO2/(0.2+SO2)*(SNH4+SNH3)/(0.2+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    B = 2.0*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*SO2/(0.2+SO2)*0.2/(0.2+SNH4+SNH3)*SNO3/(0.2+SNO3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    C = 0.2*np.exp(0.07*(T-20.0))*SO2/(0.2+SO2)*XH
    D = 1.6*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*0.2/(0.2+SO2)*SNO3/(0.5+SNO3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    E = 1.6*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*0.2/(0.2+SO2)*SNO2/(0.2+SNO2)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    F = 0.1*np.exp(0.07*(T-20.0))*0.2/(0.2+SO2)*SNO3/(0.5+SNO3)*XH
    G = 0.8*np.exp(0.098*(T-20.0))*SO2/(0.5+SO2)*(SNH4+SNH3)/(0.5+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XN1
    H = 0.05*np.exp(0.098*(T-20.0))*SO2/(0.5+SO2)*XN1
    I = 1.1*np.exp(0.069*(T-20.0))*SO2/(0.5+SO2)*SNO2/(0.5+SNO2)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XN2
    J = 0.05*np.exp(0.069*(T-20.0))*SO2/(0.5+SO2)*XN2
    K = 2.0*np.exp(0.046*(T-20.0))*(SNH4+SNH3+SNO3)/(0.1+SNH4+SNH3+SNO3)*(SNH4+SNH3)/(0.1+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*Ie/500.0*np.exp(1.0-Ie/500.0)*XALG
    LL = 2.0*np.exp(0.046*(T-20.0))*(SNH4+SNH3+SNO3)/(0.1+SNH4+SNH3+SNO3)*0.1/(0.1+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*Ie/500.0*np.exp(1.0-Ie/500.0)*XALG
    M = 0.1*np.exp(0.046*(T-20.0))*SO2/(0.2+SO2)*XALG
    N = 0.1*np.exp(0.046*(T-20.0))*XALG
    O = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XALG*XCON
    P = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XS*XCON
    Q = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XH*XCON
    R = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN1*XCON
    S = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN2*XCON
    TT = 0.05*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XCON
    U = 0.05*np.exp(0.08*(T-20.0))*XCON
    V = 0.3*np.exp(0.07*(T-20.0))*XS
    AA = 0.0001*(SH2PO4-SH*SHPO4/KeqP)
    return q/v*(SHPO4in-SHPO4)+(-0.0083*A-0.0083*B+0.017*C-0.0062*D+0.0021*E+0.017*F-0.019*G+0.017*H-0.019*I+0.017*J-0.011*K-0.011*LL+0.0086*M+0.0041*N+0.022*O+0.022*P+0.13*Q+0.13*R+0.13*S+0.0086*TT+0.0041*U+0.0*V+1.0*AA)

def f_SH2PO4(SH2PO4in, SHPO4, SH2PO4, SH, T, q, v):
    AA = 0.0001*(SH2PO4-SH*SHPO4/KeqP)
    return q/v*(SH2PO4in-SH2PO4)+(-1.0*AA)

def f_SO2(SO2in,  SS, SI, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, SH, SOH, SCa, XH, XN1, XN2, XALG, XCON, XS,  T, q, v, Ie):
    A = 2.0*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*SO2/(0.2+SO2)*(SNH4+SNH3)/(0.2+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    B = 2.0*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*SO2/(0.2+SO2)*0.2/(0.2+SNH4+SNH3)*SNO3/(0.2+SNO3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    C = 0.2*np.exp(0.07*(T-20.0))*SO2/(0.2+SO2)*XH
    G = 0.8*np.exp(0.098*(T-20.0))*SO2/(0.5+SO2)*(SNH4+SNH3)/(0.5+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XN1
    H = 0.05*np.exp(0.098*(T-20.0))*SO2/(0.5+SO2)*XN1
    I = 1.1*np.exp(0.069*(T-20.0))*SO2/(0.5+SO2)*SNO2/(0.5+SNO2)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XN2
    J = 0.05*np.exp(0.069*(T-20.0))*SO2/(0.5+SO2)*XN2
    K = 2.0*np.exp(0.046*(T-20.0))*(SNH4+SNH3+SNO3)/(0.1+SNH4+SNH3+SNO3)*(SNH4+SNH3)/(0.1+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)
    LL = 2.0*np.exp(0.046*(T-20.0))*(SNH4+SNH3+SNO3)/(0.1+SNH4+SNH3+SNO3)*0.1/(0.1+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*Ie/50
    M = 0.1*np.exp(0.046*(T-20.0))*SO2/(0.2+SO2)*XALG
    N = 0.1*np.exp(0.046*(T-20.0))*XALG
    O = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XALG*XCON
    P = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XS*XCON
    Q = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XH*XCON
    R = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN1*XCON
    S = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN2*XCON
    TT = 0.05*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XCON
    U = 0.05*np.exp(0.08*(T-20.0))*XCON
    V = 0.3*np.exp(0.07*(T-20.0))*XS
    return q/v*(SO2in-SO2)+(-0.85*A-0.80*B-0.77*C-15.0*G-0.77*H-22.0*I-0.77*J+1.0*K+1.3*LL-0.60*M +0.20*N-0.15*O-4.8*P-3.8*Q-3.8*R-3.8*S-0.60*TT+0.20*U+0.0*V)+K2*(14.652-0.41022*T+0.007991*T*T-0.000077774*T*T*T-SO2)

def f_SCO2(SCO2in, SCO2, SHCO3, SH, T, q, v):
    W = 0.001*(SCO2-SH*SHCO3/Keq1)
    return q/v*(SCO2in-SCO2)+(-1.0*W)

def f_SHCO3(SHCO3in, SS, SI, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, SH, SOH, SCa, XH, XN1, XN2, XALG, XCON, XS,  T, q, v, Ie):
    A = 2.0*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*SO2/(0.2+SO2)*(SNH4+SNH3)/(0.2+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    B = 2.0*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*SO2/(0.2+SO2)*0.2/(0.2+SNH4+SNH3)*SNO3/(0.2+SNO3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    C = 0.2*np.exp(0.07*(T-20.0))*SO2/(0.2+SO2)*XH
    D = 1.6*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*0.2/(0.2+SO2)*SNO3/(0.5+SNO3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    E = 1.6*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*0.2/(0.2+SO2)*SNO2/(0.2+SNO2)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    F = 0.1*np.exp(0.07*(T-20.0))*0.2/(0.2+SO2)*SNO3/(0.5+SNO3)*XH
    G = 0.8*np.exp(0.098*(T-20.0))*SO2/(0.5+SO2)*(SNH4+SNH3)/(0.5+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XN1
    H = 0.05*np.exp(0.098*(T-20.0))*SO2/(0.5+SO2)*XN1
    I = 1.1*np.exp(0.069*(T-20.0))*SO2/(0.5+SO2)*SNO2/(0.5+SNO2)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XN2
    J = 0.05*np.exp(0.069*(T-20.0))*SO2/(0.5+SO2)*XN2
    K = 2.0*np.exp(0.046*(T-20.0))*(SNH4+SNH3+SNO3)/(0.1+SNH4+SNH3+SNO3)*(SNH4+SNH3)/(0.1+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)
    LL = 2.0*np.exp(0.046*(T-20.0))*(SNH4+SNH3+SNO3)/(0.1+SNH4+SNH3+SNO3)*0.1/(0.1+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*Ie/50
    M = 0.1*np.exp(0.046*(T-20.0))*SO2/(0.2+SO2)*XALG
    N = 0.1*np.exp(0.046*(T-20.0))*XALG
    O = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XALG*XCON
    P = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XS*XCON
    Q = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XH*XCON
    R = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN1*XCON
    S = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN2*XCON
    TT = 0.05*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XCON
    U = 0.05*np.exp(0.08*(T-20.0))*XCON
    V = 0.3*np.exp(0.07*(T-20.0))*XS
    W = 0.001*(SCO2-SH*SHCO3/Keq1)
    X = 0.0001*(SHCO3-SH*SCO3/Keq2)
    return q/v*(SHCO3in-SHCO3)+(0.27*A+0.27*B+0.25*C+0.39*D+0.86*E+0.25*F-0.32*G+0.25*H-0.32*I+0.25*J-0.39*K-0.39*LL+0.26*M+0.0018*N+0.32*O+1.5*P+1.2*Q+1.2*R+1.2*S+0.26*TT+0.0018*U+0.0*V+1.0*W-1.0*X)

def f_SCO3(SCO3in, SHCO3, SCO3, SH, SCa, T, q, v):
    X = 0.0001*(SHCO3-SH*SCO3/Keq2)
    AB = 0.00000002*(1.0-SCa*SCO3/12/40/Keqso)
    return q/v*(SCO3in-SCO3)+(1.0*X+0.30*AB)

def f_SH(SHin, SS, SI, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, SH, SOH, SCa, XH, XN1, XN2, XALG, XCON, XS,  T, q, v, Ie):
    A = 2.0*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*SO2/(0.2+SO2)*(SNH4+SNH3)/(0.2+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    B = 2.0*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*SO2/(0.2+SO2)*0.2/(0.2+SNH4+SNH3)*SNO3/(0.2+SNO3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    C = 0.2*np.exp(0.07*(T-20.0))*SO2/(0.2+SO2)*XH
    D = 1.6*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*0.2/(0.2+SO2)*SNO3/(0.5+SNO3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    E = 1.6*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*0.2/(0.2+SO2)*SNO2/(0.2+SNO2)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    F = 0.1*np.exp(0.07*(T-20.0))*0.2/(0.2+SO2)*SNO3/(0.5+SNO3)*XH
    G = 0.8*np.exp(0.098*(T-20.0))*SO2/(0.5+SO2)*(SNH4+SNH3)/(0.5+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XN1
    H = 0.05*np.exp(0.098*(T-20.0))*SO2/(0.5+SO2)*XN1
    I = 1.1*np.exp(0.069*(T-20.0))*SO2/(0.5+SO2)*SNO2/(0.5+SNO2)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XN2
    J = 0.05*np.exp(0.069*(T-20.0))*SO2/(0.5+SO2)*XN2
    K = 2.0*np.exp(0.046*(T-20.0))*(SNH4+SNH3+SNO3)/(0.1+SNH4+SNH3+SNO3)*(SNH4+SNH3)/(0.1+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)
    LL = 2.0*np.exp(0.046*(T-20.0))*(SNH4+SNH3+SNO3)/(0.1+SNH4+SNH3+SNO3)*0.1/(0.1+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*Ie/50
    M = 0.1*np.exp(0.046*(T-20.0))*SO2/(0.2+SO2)*XALG
    N = 0.1*np.exp(0.046*(T-20.0))*XALG
    O = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XALG*XCON
    P = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XS*XCON
    Q = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XH*XCON
    R = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN1*XCON
    S = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN2*XCON
    TT = 0.05*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XCON
    U = 0.05*np.exp(0.08*(T-20.0))*XCON
    V = 0.3*np.exp(0.07*(T-20.0))*XS
    W = 0.001*(SCO2-SH*SHCO3/Keq1)
    X = 0.0001*(SHCO3-SH*SCO3/Keq2)
    Y = 0.0001*(1.0-SH*SOH/Keqw)
    Z = 0.0001*(SNH4-SH*SNH3/KeqN)
    AA = 0.0001*(SH2PO4-SH*SHPO4/KeqP)
    return q/v*(SHin-SH)+(0.023*A+0.021*B+0.017*C+0.032*D-0.045*E+0.0025*F+0.65*G+0.017*H-0.033*I+0.017*J-0.028*K-0.038*LL+0.018*M+0.0016*N+0.019*O+0.11*P+0.075*Q+0.075*R+0.075*S+0.018*TT+0.0016*U+0.0*V+0.083*W+0.083*X+1.0*Y+0.071*Z+0.032*AA)

def f_SOH(SOHin, SH, SOH, SCa, q, v):
    Y = 0.0001*(1.0-SH*SOH/Keqw)
    return q/v*(SOHin-SOH)+(1.0*Y)

def f_SCa(SCain, SCO3,SCa, q, v):
    AB = 0.00000002*(1.0-SCa*SCO3/12/40/Keqso)
    return q/v*(SCain-SCa)+(1.0*AB)

def f_XH(XHin, SS, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, XH, XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c):
    A = 2.0*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*SO2/(0.2+SO2)*(SNH4+SNH3)/(0.2+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    B = 2.0*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*SO2/(0.2+SO2)*0.2/(0.2+SNH4+SNH3)*SNO3/(0.2+SNO3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    C = 0.2*np.exp(0.07*(T-20.0))*SO2/(0.2+SO2)*XH
    D = 1.6*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*0.2/(0.2+SO2)*SNO3/(0.5+SNO3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    E = 1.6*np.exp(0.07*(T-20.0))*SS/(2.0+SS)*0.2/(0.2+SO2)*SNO2/(0.2+SNO2)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XH
    F = 0.1*np.exp(0.07*(T-20.0))*0.2/(0.2+SO2)*SNO3/(0.5+SNO3)*XH
    Q = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XH*XCON
    return q/v*(XHin-XH)+(1.0*A+1.0*B-1.0*C+1.0*D+1.0*E-1.0*F-8.7*Q)+As/v*dt*(Ma*(vb-27216.0)*(vb-27216.0)*XHin0/(XHin0+XN1in0+XN2in0+XALGin0+XCONin0+XSin0a+XSin0b+XSin0c+XIin0a+XIin0b+XIin0c)-w*XH)

def f_XN1(XN1in, SNH4, SNH3, SHPO4, SH2PO4, SO2, XN1, XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c):
    G = 0.8*np.exp(0.098*(T-20.0))*SO2/(0.5+SO2)*(SNH4+SNH3)/(0.5+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XN1
    H = 0.05*np.exp(0.098*(T-20.0))*SO2/(0.5+SO2)*XN1
    R = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN1*XCON
    return q/v*(XN1in-XN1)+(1.0*G-1.0*H-8.7*R)+As/v*dt*(Ma*(vb-27216.0)*(vb-27216.0)*XN1in0/(XHin0+XN1in0+XN2in0+XALGin0+XCONin0+XSin0a+XSin0b+XSin0c+XIin0a+XIin0b+XIin0c)-w*XN1)

def f_XN2(XN2in, SNO2, SHPO4, SH2PO4, SO2, XH, XN2, XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c):
    I = 1.1*np.exp(0.069*(T-20.0))*SO2/(0.5+SO2)*SNO2/(0.5+SNO2)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*XN2
    J = 0.05*np.exp(0.069*(T-20.0))*SO2/(0.5+SO2)*XN2
    S = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN2*XCON
    return q/v*(XN2in-XN2)+(1.0*I-1.0*J-8.7*S)+As/v*dt*(Ma*(vb-27216.0)*(vb-27216.0)*XN2in0/(XHin0+XN1in0+XN2in0+XALGin0+XCONin0+XSin0a+XSin0b+XSin0c+XIin0a+XIin0b+XIin0c)-w*XN2)

def f_XALG(XALGin, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, XALG, XCON,  T, q, v, Ie, vb,  Ma, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c):
    K = 2.0*np.exp(0.046*(T-20.0))*(SNH4+SNH3+SNO3)/(0.1+SNH4+SNH3+SNO3)*(SNH4+SNH3)/(0.1+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)
    LL = 2.0*np.exp(0.046*(T-20.0))*(SNH4+SNH3+SNO3)/(0.1+SNH4+SNH3+SNO3)*0.1/(0.1+SNH4+SNH3)*(SHPO4+SH2PO4)/(0.02+SHPO4+SH2PO4)*Ie/50
    M = 0.1*np.exp(0.046*(T-20.0))*SO2/(0.2+SO2)*XALG
    N = 0.1*np.exp(0.046*(T-20.0))*XALG
    O = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XALG*XCON
    return q/v*(XALGin-XALG)+(1.0*K+1.0*LL-1.0*M-1.0*N-5.0*O)+As/v*dt*(Ma*(vb-27216.0)*(vb-27216.0)*XALGin0/(XHin0+XN1in0+XN2in0+XALGin0+XCONin0+XSin0a+XSin0b+XSin0c+XIin0a+XIin0b+XIin0c)-w*XALG)

def f_XCON(XCONin, SO2, XH, XN1, XN2, XALG, XCON, XS,  T, q, v, Ma, vb,  XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c):
    O = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XALG*XCON
    P = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XS*XCON
    Q = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XH*XCON
    R = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN1*XCON
    S = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN2*XCON
    TT = 0.05*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XCON
    U = 0.05*np.exp(0.08*(T-20.0))*XCON
    return q/v*(XCONin-XCON)+(1.0*O+1.0*P+1.0*Q+1.0*R+1.0*S-1.0*TT-1.0*U)+As/v*dt*(Ma*(vb-27216.0)*(vb-27216.0)*XCONin0/(XHin0+XN1in0+XN2in0+XALGin0+XCONin0+XSin0a+XSin0b+XSin0c+XIin0a+XIin0b+XIin0c)-w*XCON)

def f_XS(XSin, SS, SI, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, SH, SOH, SCa, XH, XN1, XN2, XALG, XCON, XS,  T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c):
    N = 0.1*np.exp(0.046*(T-20.0))*XALG
    O = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XALG*XCON
    P = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XS*XCON
    Q = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XH*XCON
    R = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN1*XCON
    S = 0.0002*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XN2*XCON
    U = 0.05*np.exp(0.08*(T-20.0))*XCON
    V = 0.3*np.exp(0.07*(T-20.0))*XS
    return q/v*(XSin-XS)+(0.95*N+3.8*O-5.8*P+3.8*Q+3.8*R+3.8*S+0.95*U-1.0*V)+As/v*dt*(Ma*(vb-27216.0)*(vb-27216.0)*(XSin0a+XSin0b+XSin0c)/(XHin0+XN1in0+XN2in0+XALGin0+XCONin0+XSin0a+XSin0b+XSin0c+XIin0a+XIin0b+XIin0c)-w*1000*XS)

def f_XI(XIin, SNO3, SO2, XH, XN1, XN2, XALG, XCON, XS, XI,  T, q, v, vb, Ma,  XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c):
    C = 0.2*np.exp(0.07*(T-20.0))*SO2/(0.2+SO2)*XH
    F = 0.1*np.exp(0.07*(T-20.0))*0.2/(0.2+SO2)*SNO3/(0.5+SNO3)*XH
    H = 0.05*np.exp(0.098*(T-20.0))*SO2/(0.5+SO2)*XN1
    J = 0.05*np.exp(0.069*(T-20.0))*SO2/(0.5+SO2)*XN2
    M = 0.1*np.exp(0.046*(T-20.0))*SO2/(0.2+SO2)*XALG
    N = 0.1*np.exp(0.046*(T-20.0))*XALG
    TT = 0.05*np.exp(0.08*(T-20.0))*SO2/(0.5+SO2)*XCON
    U = 0.05*np.exp(0.08*(T-20.0))*XCON
    return q/v*(XIin-XI)+(0.23*C+0.23*F+0.23*H+0.23*J+0.40*M+0.25*N+0.40*TT+0.25*U)+As/v*dt*(Ma*(vb-27216.0)*(vb-27216.0)*(XIin0a+XIin0b+XIin0c)/(XHin0+XN1in0+XN2in0+XALGin0+XCONin0+XSin0a+XSin0b+XSin0c+XIin0a+XIin0b+XIin0c)-w*1000*XI)
# def f_XP(XPin, SS, SI, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, SH, SOH, SCa, XH, XN1, XN2, XALG, XCON, XS, XP):

#     return  


# establish the function "runge_kutta"
def runge_kutta(SSin, SIin, SNH4in, SNH3in, SNO2in, SNO3in, SHPO4in, SH2PO4in, SO2in, SCO2in, SHCO3in, SCO3in, SHin, SOHin, SCain, XHin, XN1in, XN2in, XALGin, XCONin, XSin, XIin, SS, SI, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, SH, SOH, SCa, XH, XN1, XN2, XALG, XCON, XS, XI, T, q, v, Ie, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c):
    

    # 1st  step; 
    rk1_SS = dt * f_SS(SSin, SS, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, XH, XS, T, q, v)
    rk1_SI = dt * f_SI(SIin, SI, q, v)
    rk1_SNH4 = dt * f_SNH4(SNH4in, SS, SI, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, SH, SOH, SCa, XH, XN1, XN2, XALG, XCON, XS, T, q, v, Ie)
    rk1_SNH3 = dt * f_SNH3(SNH3in, SNH4, SNH3, SH, T, q, v)
    rk1_SNO2 = dt * f_SNO2(SNO2in, SS, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, XH, XN1, XN2, T, q, v)
    rk1_SNO3 = dt * f_SNO3(SNO3in, SS, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, XH, XN2, XALG, T, q, v, Ie)
    rk1_SHPO4 = dt * f_SHPO4(SHPO4in, SS, SI, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, SH, SOH, SCa, XH, XN1, XN2, XALG, XCON, XS, T, q, v, Ie)
    rk1_SH2PO4 = dt * f_SH2PO4(SH2PO4in, SHPO4, SH2PO4, SH, T, q, v)
    rk1_SO2 = dt * f_SO2(SO2in, SS, SI, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, SH, SOH, SCa, XH, XN1, XN2, XALG, XCON, XS, T, q, v, Ie)
    rk1_SCO2 = dt * f_SCO2(SCO2in, SCO2, SHCO3, SH, T, q, v)
    rk1_SHCO3 = dt * f_SHCO3(SHCO3in, SS, SI, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, SH, SOH, SCa, XH, XN1, XN2, XALG, XCON, XS, T, q, v, Ie)
    rk1_SCO3 = dt * f_SCO3(SCO3in, SHCO3, SCO3, SH, SCa, T, q, v)
    rk1_SH = dt * f_SH(SHin, SS, SI, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, SH, SOH, SCa, XH, XN1, XN2, XALG, XCON, XS, T, q, v, Ie)
    rk1_SOH = dt * f_SOH(SOHin, SH, SOH, SCa, q, v)
    rk1_SCa = dt * f_SCa(SCain, SCO3, SCa, q, v)
    rk1_XH = dt * f_XH(XHin, SS, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, XH, XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk1_XN1 = dt * f_XN1(XN1in, SNH4, SNH3, SHPO4, SH2PO4, SO2, XN1, XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk1_XN2 = dt * f_XN2(XN2in, SNO2, SHPO4, SH2PO4, SO2, XH, XN2, XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk1_XALG = dt * f_XALG(XALGin, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, XALG, XCON, T, q, v, Ie, vb, Ma, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk1_XCON = dt * f_XCON(XCONin, SO2, XH, XN1, XN2, XALG, XCON, XS, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk1_XS = dt * f_XS(XSin, SS, SI, SNH4, SNH3, SNO2, SNO3, SHPO4, SH2PO4, SO2, SCO2, SHCO3, SCO3, SH, SOH, SCa, XH, XN1, XN2, XALG, XCON, XS, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk1_XI = dt * f_XI(XIin, SNO3, SO2, XH, XN1, XN2, XALG, XCON, XS, XI, T, q, v, vb, Ma, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)

    # 2nd step; add 1/2
    rk2_SS = dt * f_SS(SSin, SS+1/2*rk1_SS, SNH4+1/2*rk1_SNH4, SNH3+1/2*rk1_SNH3, SNO2+1/2*rk1_SNO2, SNO3+1/2*rk1_SNO3, SHPO4+1/2*rk1_SHPO4, SH2PO4+1/2*rk1_SH2PO4, SO2+1/2*rk1_SO2, XH+1/2*rk1_XH, XS+1/2*rk1_XS, T, q, v)
    rk2_SI = dt * f_SI(SIin, SI+1/2*rk1_SI, q, v)
    rk2_SNH4 = dt * f_SNH4(SNH4in, SS+1/2*rk1_SS, SI+1/2*rk1_SI, SNH4+1/2*rk1_SNH4, SNH3+1/2*rk1_SNH3, SNO2+1/2*rk1_SNO2, SNO3+1/2*rk1_SNO3, SHPO4+1/2*rk1_SHPO4, SH2PO4+1/2*rk1_SH2PO4, SO2+1/2*rk1_SO2, SCO2+1/2*rk1_SCO2, SHCO3+1/2*rk1_SHCO3, SCO3+1/2*rk1_SCO3, SH+1/2*rk1_SH, SOH+1/2*rk1_SOH, SCa+1/2*rk1_SCa, XH+1/2*rk1_XH, XN1+1/2*rk1_XN1, XN2+1/2*rk1_XN2, XALG+1/2*rk1_XALG, XCON+1/2*rk1_XCON, XS+1/2*rk1_XS, T, q, v, Ie)
    rk2_SNH3 = dt * f_SNH3(SNH3in, SNH4+1/2*rk1_SNH4, SNH3+1/2*rk1_SNH3, SH+1/2*rk1_SH, T, q, v)
    rk2_SNO2 = dt * f_SNO2(SNO2in, SS+1/2*rk1_SS, SNH4+1/2*rk1_SNH4, SNH3+1/2*rk1_SNH3, SNO2+1/2*rk1_SNO2, SNO3+1/2*rk1_SNO3, SHPO4+1/2*rk1_SHPO4, SH2PO4+1/2*rk1_SH2PO4, SO2+1/2*rk1_SO2, XH+1/2*rk1_XH, XN1+1/2*rk1_XN1, XN2+1/2*rk1_XN2, T, q, v)
    rk2_SNO3 = dt * f_SNO3(SNO3in, SS+1/2*rk1_SS, SNH4+1/2*rk1_SNH4, SNH3+1/2*rk1_SNH3, SNO2+1/2*rk1_SNO2, SNO3+1/2*rk1_SNO3, SHPO4+1/2*rk1_SHPO4, SH2PO4+1/2*rk1_SH2PO4, SO2+1/2*rk1_SO2, XH+1/2*rk1_XH, XN2+1/2*rk1_XN2, XALG+1/2*rk1_XALG, T, q, v, Ie)
    rk2_SHPO4 = dt * f_SHPO4(SHPO4in, SS+1/2*rk1_SS, SI+1/2*rk1_SI, SNH4+1/2*rk1_SNH4, SNH3+1/2*rk1_SNH3, SNO2+1/2*rk1_SNO2, SNO3+1/2*rk1_SNO3, SHPO4+1/2*rk1_SHPO4, SH2PO4+1/2*rk1_SH2PO4, SO2+1/2*rk1_SO2, SCO2+1/2*rk1_SCO2, SHCO3+1/2*rk1_SHCO3, SCO3+1/2*rk1_SCO3, SH+1/2*rk1_SH, SOH+1/2*rk1_SOH, SCa+1/2*rk1_SCa, XH+1/2*rk1_XH, XN1+1/2*rk1_XN1, XN2+1/2*rk1_XN2, XALG+1/2*rk1_XALG, XCON+1/2*rk1_XCON, XS+1/2*rk1_XS, T, q, v, Ie)
    rk2_SH2PO4 = dt * f_SH2PO4(SH2PO4in, SHPO4+1/2*rk1_SHPO4, SH2PO4+1/2*rk1_SH2PO4, SH+1/2*rk1_SH, T, q, v)
    rk2_SO2 = dt * f_SO2(SO2in, SS+1/2*rk1_SS, SI+1/2*rk1_SI, SNH4+1/2*rk1_SNH4, SNH3+1/2*rk1_SNH3, SNO2+1/2*rk1_SNO2, SNO3+1/2*rk1_SNO3, SHPO4+1/2*rk1_SHPO4, SH2PO4+1/2*rk1_SH2PO4, SO2+1/2*rk1_SO2, SCO2+1/2*rk1_SCO2, SHCO3+1/2*rk1_SHCO3, SCO3+1/2*rk1_SCO3, SH+1/2*rk1_SH, SOH+1/2*rk1_SOH, SCa+1/2*rk1_SCa, XH+1/2*rk1_XH, XN1+1/2*rk1_XN1, XN2+1/2*rk1_XN2, XALG+1/2*rk1_XALG, XCON+1/2*rk1_XCON, XS+1/2*rk1_XS, T, q, v, Ie)
    rk2_SCO2 = dt * f_SCO2(SCO2in, SCO2+1/2*rk1_SCO2, SHCO3+1/2*rk1_SHCO3, SH+1/2*rk1_SH, T, q, v)
    rk2_SHCO3 = dt * f_SHCO3(SHCO3in, SS+1/2*rk1_SS, SI+1/2*rk1_SI, SNH4+1/2*rk1_SNH4, SNH3+1/2*rk1_SNH3, SNO2+1/2*rk1_SNO2, SNO3+1/2*rk1_SNO3, SHPO4+1/2*rk1_SHPO4, SH2PO4+1/2*rk1_SH2PO4, SO2+1/2*rk1_SO2, SCO2+1/2*rk1_SCO2, SHCO3+1/2*rk1_SHCO3, SCO3+1/2*rk1_SCO3, SH+1/2*rk1_SH, SOH+1/2*rk1_SOH, SCa+1/2*rk1_SCa, XH+1/2*rk1_XH, XN1+1/2*rk1_XN1, XN2+1/2*rk1_XN2, XALG+1/2*rk1_XALG, XCON+1/2*rk1_XCON, XS+1/2*rk1_XS, T, q, v, Ie)
    rk2_SCO3 = dt * f_SCO3(SCO3in, SHCO3+1/2*rk1_SHCO3, SCO3+1/2*rk1_SCO3, SH+1/2*rk1_SH, SCa+1/2*rk1_SCa, T, q, v)
    rk2_SH = dt * f_SH(SHin, SS+1/2*rk1_SS, SI+1/2*rk1_SI, SNH4+1/2*rk1_SNH4, SNH3+1/2*rk1_SNH3, SNO2+1/2*rk1_SNO2, SNO3+1/2*rk1_SNO3, SHPO4+1/2*rk1_SHPO4, SH2PO4+1/2*rk1_SH2PO4, SO2+1/2*rk1_SO2, SCO2+1/2*rk1_SCO2, SHCO3+1/2*rk1_SHCO3, SCO3+1/2*rk1_SCO3, SH+1/2*rk1_SH, SOH+1/2*rk1_SOH, SCa+1/2*rk1_SCa, XH+1/2*rk1_XH, XN1+1/2*rk1_XN1, XN2+1/2*rk1_XN2, XALG+1/2*rk1_XALG, XCON+1/2*rk1_XCON, XS+1/2*rk1_XS, T, q, v, Ie)
    rk2_SOH = dt * f_SOH(SOHin, SH+1/2*rk1_SH, SOH+1/2*rk1_SOH, SCa+1/2*rk1_SCa, q, v)
    rk2_SCa = dt * f_SCa(SCain, SCO3+1/2*rk1_SCO3, SCa+1/2*rk1_SCa, q, v)
    rk2_XH = dt * f_XH(XHin, SS+1/2*rk1_SS, SNH4+1/2*rk1_SNH4, SNH3+1/2*rk1_SNH3, SNO2+1/2*rk1_SNO2, SNO3+1/2*rk1_SNO3, SHPO4+1/2*rk1_SHPO4, SH2PO4+1/2*rk1_SH2PO4, SO2+1/2*rk1_SO2, XH+1/2*rk1_XH, XCON+1/2*rk1_XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk2_XN1 = dt * f_XN1(XN1in, SNH4+1/2*rk1_SNH4, SNH3+1/2*rk1_SNH3, SHPO4+1/2*rk1_SHPO4, SH2PO4+1/2*rk1_SH2PO4, SO2+1/2*rk1_SO2, XN1+1/2*rk1_XN1, XCON+1/2*rk1_XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk2_XN2 = dt * f_XN2(XN2in, SNO2+1/2*rk1_SNO2, SHPO4+1/2*rk1_SHPO4, SH2PO4+1/2*rk1_SH2PO4, SO2+1/2*rk1_SO2, XH+1/2*rk1_XH, XN2+1/2*rk1_XN2, XCON+1/2*rk1_XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk2_XALG = dt * f_XALG(XALGin, SNH4+1/2*rk1_SNH4, SNH3+1/2*rk1_SNH3, SNO2+1/2*rk1_SNO2, SNO3+1/2*rk1_SNO3, SHPO4+1/2*rk1_SHPO4, SH2PO4+1/2*rk1_SH2PO4, SO2+1/2*rk1_SO2, SCO2+1/2*rk1_SCO2, SHCO3+1/2*rk1_SHCO3, SCO3+1/2*rk1_SCO3, XALG+1/2*rk1_XALG, XCON+1/2*rk1_XCON, T, q, v, Ie, vb, Ma, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk2_XCON = dt * f_XCON(XCONin, SO2+1/2*rk1_SO2, XH+1/2*rk1_XH, XN1+1/2*rk1_XN1, XN2+1/2*rk1_XN2, XALG+1/2*rk1_XALG, XCON+1/2*rk1_XCON, XS+1/2*rk1_XS, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk2_XS = dt * f_XS(XSin, SS+1/2*rk1_SS, SI+1/2*rk1_SI, SNH4+1/2*rk1_SNH4, SNH3+1/2*rk1_SNH3, SNO2+1/2*rk1_SNO2, SNO3+1/2*rk1_SNO3, SHPO4+1/2*rk1_SHPO4, SH2PO4+1/2*rk1_SH2PO4, SO2+1/2*rk1_SO2, SCO2+1/2*rk1_SCO2, SHCO3+1/2*rk1_SHCO3, SCO3+1/2*rk1_SCO3, SH+1/2*rk1_SH, SOH+1/2*rk1_SOH, SCa+1/2*rk1_SCa, XH+1/2*rk1_XH, XN1+1/2*rk1_XN1, XN2+1/2*rk1_XN2, XALG+1/2*rk1_XALG, XCON+1/2*rk1_XCON, XS+1/2*rk1_XS, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk2_XI = dt * f_XI(XIin, SNO3+1/2*rk1_SNO3, SO2+1/2*rk1_SO2, XH+1/2*rk1_XH, XN1+1/2*rk1_XN1, XN2+1/2*rk1_XN2, XALG+1/2*rk1_XALG, XCON+1/2*rk1_XCON, XS+1/2*rk1_XS, XI+1/2*rk1_XI, T, q, v, vb, Ma, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)

    # 3rd step; add 1/2
    rk3_SS = dt * f_SS(SSin, SS+1/2*rk2_SS, SNH4+1/2*rk2_SNH4, SNH3+1/2*rk2_SNH3, SNO2+1/2*rk2_SNO2, SNO3+1/2*rk2_SNO3, SHPO4+1/2*rk2_SHPO4, SH2PO4+1/2*rk2_SH2PO4, SO2+1/2*rk2_SO2, XH+1/2*rk2_XH, XS+1/2*rk2_XS, T, q, v)
    rk3_SI = dt * f_SI(SIin, SI+1/2*rk2_SI, q, v)
    rk3_SNH4 = dt * f_SNH4(SNH4in, SS+1/2*rk2_SS, SI+1/2*rk2_SI, SNH4+1/2*rk2_SNH4, SNH3+1/2*rk2_SNH3, SNO2+1/2*rk2_SNO2, SNO3+1/2*rk2_SNO3, SHPO4+1/2*rk2_SHPO4, SH2PO4+1/2*rk2_SH2PO4, SO2+1/2*rk2_SO2, SCO2+1/2*rk2_SCO2, SHCO3+1/2*rk2_SHCO3, SCO3+1/2*rk2_SCO3, SH+1/2*rk2_SH, SOH+1/2*rk2_SOH, SCa+1/2*rk2_SCa, XH+1/2*rk2_XH, XN1+1/2*rk2_XN1, XN2+1/2*rk2_XN2, XALG+1/2*rk2_XALG, XCON+1/2*rk2_XCON, XS+1/2*rk2_XS, T, q, v, Ie)
    rk3_SNH3 = dt * f_SNH3(SNH3in, SNH4+1/2*rk2_SNH4, SNH3+1/2*rk2_SNH3, SH+1/2*rk2_SH, T, q, v)
    rk3_SNO2 = dt * f_SNO2(SNO2in, SS+1/2*rk2_SS, SNH4+1/2*rk2_SNH4, SNH3+1/2*rk2_SNH3, SNO2+1/2*rk2_SNO2, SNO3+1/2*rk2_SNO3, SHPO4+1/2*rk2_SHPO4, SH2PO4+1/2*rk2_SH2PO4, SO2+1/2*rk2_SO2, XH+1/2*rk2_XH, XN1+1/2*rk2_XN1, XN2+1/2*rk2_XN2, T, q, v)
    rk3_SNO3 = dt * f_SNO3(SNO3in, SS+1/2*rk2_SS, SNH4+1/2*rk2_SNH4, SNH3+1/2*rk2_SNH3, SNO2+1/2*rk2_SNO2, SNO3+1/2*rk2_SNO3, SHPO4+1/2*rk2_SHPO4, SH2PO4+1/2*rk2_SH2PO4, SO2+1/2*rk2_SO2, XH+1/2*rk2_XH, XN2+1/2*rk2_XN2, XALG+1/2*rk2_XALG, T, q, v, Ie)
    rk3_SHPO4 = dt * f_SHPO4(SHPO4in, SS+1/2*rk2_SS, SI+1/2*rk2_SI, SNH4+1/2*rk2_SNH4, SNH3+1/2*rk2_SNH3, SNO2+1/2*rk2_SNO2, SNO3+1/2*rk2_SNO3, SHPO4+1/2*rk2_SHPO4, SH2PO4+1/2*rk2_SH2PO4, SO2+1/2*rk2_SO2, SCO2+1/2*rk2_SCO2, SHCO3+1/2*rk2_SHCO3, SCO3+1/2*rk2_SCO3, SH+1/2*rk2_SH, SOH+1/2*rk2_SOH, SCa+1/2*rk2_SCa, XH+1/2*rk2_XH, XN1+1/2*rk2_XN1, XN2+1/2*rk2_XN2, XALG+1/2*rk2_XALG, XCON+1/2*rk2_XCON, XS+1/2*rk2_XS, T, q, v, Ie)
    rk3_SH2PO4 = dt * f_SH2PO4(SH2PO4in, SHPO4+1/2*rk2_SHPO4, SH2PO4+1/2*rk2_SH2PO4, SH+1/2*rk2_SH, T, q, v)
    rk3_SO2 = dt * f_SO2(SO2in, SS+1/2*rk2_SS, SI+1/2*rk2_SI, SNH4+1/2*rk2_SNH4, SNH3+1/2*rk2_SNH3, SNO2+1/2*rk2_SNO2, SNO3+1/2*rk2_SNO3, SHPO4+1/2*rk2_SHPO4, SH2PO4+1/2*rk2_SH2PO4, SO2+1/2*rk2_SO2, SCO2+1/2*rk2_SCO2, SHCO3+1/2*rk2_SHCO3, SCO3+1/2*rk2_SCO3, SH+1/2*rk2_SH, SOH+1/2*rk2_SOH, SCa+1/2*rk2_SCa, XH+1/2*rk2_XH, XN1+1/2*rk2_XN1, XN2+1/2*rk2_XN2, XALG+1/2*rk2_XALG, XCON+1/2*rk2_XCON, XS+1/2*rk2_XS, T, q, v, Ie)
    rk3_SCO2 = dt * f_SCO2(SCO2in, SCO2+1/2*rk2_SCO2, SHCO3+1/2*rk2_SHCO3, SH+1/2*rk2_SH, T, q, v)
    rk3_SHCO3 = dt * f_SHCO3(SHCO3in, SS+1/2*rk2_SS, SI+1/2*rk2_SI, SNH4+1/2*rk2_SNH4, SNH3+1/2*rk2_SNH3, SNO2+1/2*rk2_SNO2, SNO3+1/2*rk2_SNO3, SHPO4+1/2*rk2_SHPO4, SH2PO4+1/2*rk2_SH2PO4, SO2+1/2*rk2_SO2, SCO2+1/2*rk2_SCO2, SHCO3+1/2*rk2_SHCO3, SCO3+1/2*rk2_SCO3, SH+1/2*rk2_SH, SOH+1/2*rk2_SOH, SCa+1/2*rk2_SCa, XH+1/2*rk2_XH, XN1+1/2*rk2_XN1, XN2+1/2*rk2_XN2, XALG+1/2*rk2_XALG, XCON+1/2*rk2_XCON, XS+1/2*rk2_XS, T, q, v, Ie)
    rk3_SCO3 = dt * f_SCO3(SCO3in, SHCO3+1/2*rk2_SHCO3, SCO3+1/2*rk2_SCO3, SH+1/2*rk2_SH, SCa+1/2*rk2_SCa, T, q, v)
    rk3_SH = dt * f_SH(SHin, SS+1/2*rk2_SS, SI+1/2*rk2_SI, SNH4+1/2*rk2_SNH4, SNH3+1/2*rk2_SNH3, SNO2+1/2*rk2_SNO2, SNO3+1/2*rk2_SNO3, SHPO4+1/2*rk2_SHPO4, SH2PO4+1/2*rk2_SH2PO4, SO2+1/2*rk2_SO2, SCO2+1/2*rk2_SCO2, SHCO3+1/2*rk2_SHCO3, SCO3+1/2*rk2_SCO3, SH+1/2*rk2_SH, SOH+1/2*rk2_SOH, SCa+1/2*rk2_SCa, XH+1/2*rk2_XH, XN1+1/2*rk2_XN1, XN2+1/2*rk2_XN2, XALG+1/2*rk2_XALG, XCON+1/2*rk2_XCON, XS+1/2*rk2_XS, T, q, v, Ie)
    rk3_SOH = dt * f_SOH(SOHin, SH+1/2*rk2_SH, SOH+1/2*rk2_SOH, SCa+1/2*rk2_SCa, q, v)
    rk3_SCa = dt * f_SCa(SCain, SCO3+1/2*rk2_SCO3, SCa+1/2*rk2_SCa, q, v)
    rk3_XH = dt * f_XH(XHin, SS+1/2*rk2_SS, SNH4+1/2*rk2_SNH4, SNH3+1/2*rk2_SNH3, SNO2+1/2*rk2_SNO2, SNO3+1/2*rk2_SNO3, SHPO4+1/2*rk2_SHPO4, SH2PO4+1/2*rk2_SH2PO4, SO2+1/2*rk2_SO2, XH+1/2*rk2_XH, XCON+1/2*rk2_XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk3_XN1 = dt * f_XN1(XN1in, SNH4+1/2*rk2_SNH4, SNH3+1/2*rk2_SNH3, SHPO4+1/2*rk2_SHPO4, SH2PO4+1/2*rk2_SH2PO4, SO2+1/2*rk2_SO2, XN1+1/2*rk2_XN1, XCON+1/2*rk2_XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk3_XN2 = dt * f_XN2(XN2in, SNO2+1/2*rk2_SNO2, SHPO4+1/2*rk2_SHPO4, SH2PO4+1/2*rk2_SH2PO4, SO2+1/2*rk2_SO2, XH+1/2*rk2_XH, XN2+1/2*rk2_XN2, XCON+1/2*rk2_XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk3_XALG = dt * f_XALG(XALGin, SNH4+1/2*rk2_SNH4, SNH3+1/2*rk2_SNH3, SNO2+1/2*rk2_SNO2, SNO3+1/2*rk2_SNO3, SHPO4+1/2*rk2_SHPO4, SH2PO4+1/2*rk2_SH2PO4, SO2+1/2*rk2_SO2, SCO2+1/2*rk2_SCO2, SHCO3+1/2*rk2_SHCO3, SCO3+1/2*rk2_SCO3, XALG+1/2*rk2_XALG, XCON+1/2*rk2_XCON, T, q, v, Ie, vb, Ma, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk3_XCON = dt * f_XCON(XCONin, SO2+1/2*rk2_SO2, XH+1/2*rk2_XH, XN1+1/2*rk2_XN1, XN2+1/2*rk2_XN2, XALG+1/2*rk2_XALG, XCON+1/2*rk2_XCON, XS+1/2*rk2_XS, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk3_XS = dt * f_XS(XSin, SS+1/2*rk2_SS, SI+1/2*rk2_SI, SNH4+1/2*rk2_SNH4, SNH3+1/2*rk2_SNH3, SNO2+1/2*rk2_SNO2, SNO3+1/2*rk2_SNO3, SHPO4+1/2*rk2_SHPO4, SH2PO4+1/2*rk2_SH2PO4, SO2+1/2*rk2_SO2, SCO2+1/2*rk2_SCO2, SHCO3+1/2*rk2_SHCO3, SCO3+1/2*rk2_SCO3, SH+1/2*rk2_SH, SOH+1/2*rk2_SOH, SCa+1/2*rk2_SCa, XH+1/2*rk2_XH, XN1+1/2*rk2_XN1, XN2+1/2*rk2_XN2, XALG+1/2*rk2_XALG, XCON+1/2*rk2_XCON, XS+1/2*rk2_XS, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk3_XI = dt * f_XI(XIin, SNO3+1/2*rk2_SNO3, SO2+1/2*rk2_SO2, XH+1/2*rk2_XH, XN1+1/2*rk2_XN1, XN2+1/2*rk2_XN2, XALG+1/2*rk2_XALG, XCON+1/2*rk2_XCON, XS+1/2*rk2_XS, XI+1/2*rk2_XI, T, q, v, vb, Ma, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)

    # 4th step
    rk4_SS = dt * f_SS(SSin, SS+rk3_SS, SNH4+rk3_SNH4, SNH3+rk3_SNH3, SNO2+rk3_SNO2, SNO3+rk3_SNO3, SHPO4+rk3_SHPO4, SH2PO4+rk3_SH2PO4, SO2+rk3_SO2, XH+rk3_XH, XS+rk3_XS, T, q, v)
    rk4_SI = dt * f_SI(SIin, SI+rk3_SI, q, v)
    rk4_SHN4 = dt * f_SNH4(SNH4in, SS+rk3_SS, SI+rk3_SI, SNH4+rk3_SNH4, SNH3+rk3_SNH3, SNO2+rk3_SNO2, SNO3+rk3_SNO3, SHPO4+rk3_SHPO4, SH2PO4+rk3_SH2PO4, SO2+rk3_SO2, SCO2+rk3_SCO2, SHCO3+rk3_SHCO3, SCO3+rk3_SCO3, SH+rk3_SH, SOH+rk3_SOH, SCa+rk3_SCa, XH+rk3_XH, XN1+rk3_XN1, XN2+rk3_XN2, XALG+rk3_XALG, XCON+rk3_XCON, XS+rk3_XS, T, q, v, Ie)
    rk4_SNH3 = dt * f_SNH3(SNH3in, SNH4+rk3_SNH4, SNH3+rk3_SNH3, SH+rk3_SH, T, q, v)
    rk4_SNO2 = dt * f_SNO2(SNO2in, SS+rk3_SS, SNH4+rk3_SNH4, SNH3+rk3_SNH3, SNO2+rk3_SNO2, SNO3+rk3_SNO3, SHPO4+rk3_SHPO4, SH2PO4+rk3_SH2PO4, SO2+rk3_SO2, XH+rk3_XH, XN1+rk3_XN1, XN2+rk3_XN2, T, q, v)
    rk4_SNO3 = dt * f_SNO3(SNO3in, SS+rk3_SS, SNH4+rk3_SNH4, SNH3+rk3_SNH3, SNO2+rk3_SNO2, SNO3+rk3_SNO3, SHPO4+rk3_SHPO4, SH2PO4+rk3_SH2PO4, SO2+rk3_SO2, XH+rk3_XH, XN2+rk3_XN2, XALG+rk3_XALG, T, q, v, Ie)
    rk4_SHPO4 = dt * f_SHPO4(SHPO4in, SS+rk3_SS, SI+rk3_SI, SNH4+rk3_SNH4, SNH3+rk3_SNH3, SNO2+rk3_SNO2, SNO3+rk3_SNO3, SHPO4+rk3_SHPO4, SH2PO4+rk3_SH2PO4, SO2+rk3_SO2, SCO2+rk3_SCO2, SHCO3+rk3_SHCO3, SCO3+rk3_SCO3, SH+rk3_SH, SOH+rk3_SOH, SCa+rk3_SCa, XH+rk3_XH, XN1+rk3_XN1, XN2+rk3_XN2, XALG+rk3_XALG, XCON+rk3_XCON, XS+rk3_XS, T, q, v, Ie)
    rk4_SH2PO4 = dt * f_SH2PO4(SH2PO4in, SHPO4+rk3_SHPO4, SH2PO4+rk3_SH2PO4, SH+rk3_SH, T, q, v)
    rk4_SO2 = dt * f_SO2(SO2in, SS+rk3_SS, SI+rk3_SI, SNH4+rk3_SNH4, SNH3+rk3_SNH3, SNO2+rk3_SNO2, SNO3+rk3_SNO3, SHPO4+rk3_SHPO4, SH2PO4+rk3_SH2PO4, SO2+rk3_SO2, SCO2+rk3_SCO2, SHCO3+rk3_SHCO3, SCO3+rk3_SCO3, SH+rk3_SH, SOH+rk3_SOH, SCa+rk3_SCa, XH+rk3_XH, XN1+rk3_XN1, XN2+rk3_XN2, XALG+rk3_XALG, XCON+rk3_XCON, XS+rk3_XS, T, q, v, Ie)
    rk4_SCO2 = dt * f_SCO2(SCO2in, SCO2+rk3_SCO2, SHCO3+rk3_SHCO3, SH+rk3_SH, T, q, v)
    rk4_SHCO3 = dt * f_SHCO3(SHCO3in, SS+rk3_SS, SI+rk3_SI, SNH4+rk3_SNH4, SNH3+rk3_SNH3, SNO2+rk3_SNO2, SNO3+rk3_SNO3, SHPO4+rk3_SHPO4, SH2PO4+rk3_SH2PO4, SO2+rk3_SO2, SCO2+rk3_SCO2, SHCO3+rk3_SHCO3, SCO3+rk3_SCO3, SH+rk3_SH, SOH+rk3_SOH, SCa+rk3_SCa, XH+rk3_XH, XN1+rk3_XN1, XN2+rk3_XN2, XALG+rk3_XALG, XCON+rk3_XCON, XS+rk3_XS, T, q, v, Ie)
    rk4_SCO3 = dt * f_SCO3(SCO3in, SHCO3+rk3_SHCO3, SCO3+rk3_SCO3, SH+rk3_SH, SCa+rk3_SCa, T, q, v)
    rk4_SH = dt * f_SH(SHin, SS+rk3_SS, SI+rk3_SI, SNH4+rk3_SNH4, SNH3+rk3_SNH3, SNO2+rk3_SNO2, SNO3+rk3_SNO3, SHPO4+rk3_SHPO4, SH2PO4+rk3_SH2PO4, SO2+rk3_SO2, SCO2+rk3_SCO2, SHCO3+rk3_SHCO3, SCO3+rk3_SCO3, SH+rk3_SH, SOH+rk3_SOH, SCa+rk3_SCa, XH+rk3_XH, XN1+rk3_XN1, XN2+rk3_XN2, XALG+rk3_XALG, XCON+rk3_XCON, XS+rk3_XS, T, q, v, Ie)
    rk4_SOH = dt * f_SOH(SOHin, SH+rk3_SH, SOH+rk3_SOH, SCa+rk3_SCa, q, v)
    rk4_SCa = dt * f_SCa(SCain, SCO3+rk3_SCO3, SCa+rk3_SCa, q, v)
    rk4_XH = dt * f_XH(XHin, SS+rk3_SS, SNH4+rk3_SNH4, SNH3+rk3_SNH3, SNO2+rk3_SNO2, SNO3+rk3_SNO3, SHPO4+rk3_SHPO4, SH2PO4+rk3_SH2PO4, SO2+rk3_SO2, XH+rk3_XH, XCON+rk3_XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk4_XN1 = dt * f_XN1(XN1in, SNH4+rk3_SNH4, SNH3+rk3_SNH3, SHPO4+rk3_SHPO4, SH2PO4+rk3_SH2PO4, SO2+rk3_SO2, XN1+rk3_XN1, XCON+rk3_XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk4_XN2 = dt * f_XN2(XN2in, SNO2+rk3_SNO2, SHPO4+rk3_SHPO4, SH2PO4+rk3_SH2PO4, SO2+rk3_SO2, XH+rk3_XH, XN2+rk3_XN2, XCON+rk3_XCON, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk4_XALG = dt * f_XALG(XALGin, SNH4+rk3_SNH4, SNH3+rk3_SNH3, SNO2+rk3_SNO2, SNO3+rk3_SNO3, SHPO4+rk3_SHPO4, SH2PO4+rk3_SH2PO4, SO2+rk3_SO2, SCO2+rk3_SCO2, SHCO3+rk3_SHCO3, SCO3+rk3_SCO3, XALG+rk3_XALG, XCON+rk3_XCON, T, q, v, Ie, vb, Ma, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk4_XCON = dt * f_XCON(XCONin, SO2+rk3_SO2, XH+rk3_XH, XN1+rk3_XN1, XN2+rk3_XN2, XALG+rk3_XALG, XCON+rk3_XCON, XS+rk3_XS, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk4_XS = dt * f_XS(XSin, SS+rk3_SS, SI+rk3_SI, SNH4+rk3_SNH4, SNH3+rk3_SNH3, SNO2+rk3_SNO2, SNO3+rk3_SNO3, SHPO4+rk3_SHPO4, SH2PO4+rk3_SH2PO4, SO2+rk3_SO2, SCO2+rk3_SCO2, SHCO3+rk3_SHCO3, SCO3+rk3_SCO3, SH+rk3_SH, SOH+rk3_SOH, SCa+rk3_SCa, XH+rk3_XH, XN1+rk3_XN1, XN2+rk3_XN2, XALG+rk3_XALG, XCON+rk3_XCON, XS+rk3_XS, T, q, v, Ma, vb, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    rk4_XI = dt * f_XI(XIin, SNO3+rk3_SNO3, SO2+rk3_SO2, XH+rk3_XH, XN1+rk3_XN1, XN2+rk3_XN2, XALG+rk3_XALG, XCON+rk3_XCON, XS+rk3_XS, XI+rk3_XI, T, q, v, vb, Ma, XHin0, XN1in0, XN2in0, XALGin0, XCONin0, XSin0a, XSin0b, XSin0c, XIin0a, XIin0b, XIin0c)
    
    # conclution
    # rk = 1/6 * (1st + 2*2nd + 2*3rd + 4th)
    rk_SS = 1/6 * (rk1_SS + 2*rk2_SS + 2*rk3_SS + rk4_SS)
    rk_SI = 1/6 * (rk1_SI + 2*rk2_SI + 2*rk3_SI + rk4_SI)
    rk_SNH4 = 1/6 * (rk1_SNH4 + 2*rk2_SNH4 + 2*rk3_SNH4 + rk4_SHN4)
    rk_SNH3 = 1/6 * (rk1_SNH3 + 2*rk2_SNH3 + 2*rk3_SNH3 + rk4_SNH3)
    rk_SNO2 = 1/6 * (rk1_SNO2 + 2*rk2_SNO2 + 2*rk3_SNO2 + rk4_SNO2)
    rk_SNO3 = 1/6 * (rk1_SNO3 + 2*rk2_SNO3 + 2*rk3_SNO3 + rk4_SNO3)
    rk_SHPO4 = 1/6 * (rk1_SHPO4 + 2*rk2_SHPO4 + 2*rk3_SHPO4 + rk4_SHPO4)
    rk_SH2PO4 = 1/6 * (rk1_SH2PO4 + 2*rk2_SH2PO4 + 2*rk3_SH2PO4 + rk4_SH2PO4)
    rk_SO2 = 1/6 * (rk1_SO2 + 2*rk2_SO2 + 2*rk3_SO2 + rk4_SO2)
    rk_SCO2 = 1/6 * (rk1_SCO2 + 2*rk2_SCO2 + 2*rk3_SCO2 + rk4_SCO2)
    rk_SHCO3 = 1/6 * (rk1_SHCO3 + 2*rk2_SHCO3 + 2*rk3_SHCO3 + rk4_SHCO3)
    rk_SCO3 = 1/6 * (rk1_SCO3 + 2*rk2_SCO3 + 2*rk3_SCO3 + rk4_SCO3)
    rk_SH = 1/6 * (rk1_SH + 2*rk2_SH + 2*rk3_SH + rk4_SH)
    rk_SOH = 1/6 * (rk1_SOH + 2*rk2_SOH + 2*rk3_SOH + rk4_SOH)
    rk_SCa = 1/6 * (rk1_SCa + 2*rk2_SCa + 2*rk3_SCa + rk4_SCa)
    rk_XH = 1/6 * (rk1_XH + 2*rk2_XH + 2*rk3_XH + rk4_XH)
    rk_XN1 = 1/6 * (rk1_XN1 + 2*rk2_XN1 + 2*rk3_XN1 + rk4_XN1)
    rk_XN2 = 1/6 * (rk1_XN2 + 2*rk2_XN2 + 2*rk3_XN2 + rk4_XN2)
    rk_XALG = 1/6 * (rk1_XALG + 2*rk2_XALG + 2*rk3_XALG + rk4_XALG)
    rk_XCON = 1/6 * (rk1_XCON + 2*rk2_XCON + 2*rk3_XCON + rk4_XCON)
    rk_XS = 1/6 * (rk1_XS + 2*rk2_XS + 2*rk3_XS + rk4_XS)
    rk_XI = 1/6 * (rk1_XI + 2*rk2_XI + 2*rk3_XI + rk4_XI)

    
    rk_list = [rk_SS, rk_SI, rk_SNH4, rk_SNH3, rk_SNO2, rk_SNO3, rk_SHPO4, rk_SH2PO4, rk_SO2, rk_SCO2, rk_SHCO3, rk_SCO3, rk_SH, rk_SOH, rk_SCa, rk_XH, rk_XN1, rk_XN2, rk_XALG, rk_XCON, rk_XS, rk_XI]
    return rk_list


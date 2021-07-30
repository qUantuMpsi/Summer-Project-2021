import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
from numba import jit
#importing required packages
N= 53 #N dimension of matrix
#experimental parameters
Vgs= 1.61821e8
Vgi= 1.69947e8
Vgp= 1.65513e8

kp0= 1.42550e7
ki0= 7.02940e6
ks0= 7.36158e6
L=  0.005

@jit(nopython=True)
def Ffunc(eps,qs,qi):
    p = 2
    VgsR = Vgs/Vgp
    VgiR = Vgi/Vgp
    kst= 0.05*ks0
    kit= 0.05*ki0
    return np.sinc((L/(np.pi*2))*(kst+eps*kit- (VgsR*kst+ VgiR*eps*kit) - 0.5*(((qs**2)/ks0)+((qi**2)/ki0)- (((qs+qi)**2)/kp0))))*np.exp(-(abs(qs + qi)**2)/(2*p**2))

@jit(nopython=True)
def Gfunc(eps,qs,qi):
    c = 15.
    return (10/c)*np.exp(-(qs**2+qi**2)/(2*c**2))*Ffunc(eps,qs,qi)

for gamma in np.linspace(-1,1,9):
    fig=plt.figure()
    plt.subplot(1,2,1)
    G = np.identity(N)
    for i in range (N):
        for j in range(N):
            a_s= i-(N-1)/2
            a_i= j-(N-1)/2
            G[i][j]= Gfunc(gamma, a_s, a_i)
            print(i,j,G[i][j])
    
    plt.contourf(np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), abs(G))
    plt.colorbar()
    plt.plot(np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),'w--', linewidth=0.7)
    plt.plot(np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),-np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),'w--', linewidth=0.7)
    plt.title('JTMA $\gamma = %.2f$' % gamma)

    plt.subplot(1,2,2)
    def Pr(qs, qi, a_s, a_i):
        if qs>a_s and qi>a_i or qs<a_s and qi<a_i:
            return Gfunc(gamma, qs, qi)
        else:
            return -Gfunc(gamma, qs, qi)

    Pr_M= np.identity(N)
    for i in range(N):
        for j in range(N):
            a_i = i - (N - 1) / 2.
            a_s = j - (N - 1) / 2.
            I1 = sci.dblquad(Pr,-np.inf,0.,-np.inf,0.,args=[a_s,a_i])[0]
            I2 = sci.dblquad(Pr,-np.inf,0,0.,np.inf,args=[a_s,a_i])[0]
            I3 = sci.dblquad(Pr,0.,np.inf,-np.inf,0.,args=[a_s,a_i])[0]
            I4 = sci.dblquad(Pr,0.,np.inf,0.,np.inf,args=[a_s,a_i])[0]
            Pr_M[i][j] = abs(I1+I2+I3+I4) ** 2
            print(i, j, Pr_M[i][j])

    plt.contourf(np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), Pr_M)
    plt.colorbar()
    plt.xlabel('$a_s$')
    plt.ylabel('$a_i$')
    plt.title('Pr matrix $\gamma = %.2f$' % gamma)
    plt.show()

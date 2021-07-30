import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
from numba import jit
#importing modules
N= 53 #dimension of matrix
k1=1 #parameter1 =ks0/ki0
k2=1 #parameter2 =Vgi/Vgs
k3=1 #parameter3 proportional to ki_tilde/ks_tilde 

@jit(nopython=True)
def rindex(x): #function to find refractive index for ppKTP crystal 
    return (3.29100 + 0.04140/(x**2-0.03978)+9.35522/(x**2-31.45571))**.5

@jit(nopython=True)
def Ffunc(qs,qi): #function for JTMA
    p = 2 #sigma_p
    c = 16. #sigma_c
    s = 2.5 * c #sigma_s
    ls= 1.550 #wavelength of signal photons in micro m
    lp= 0.775 #wavelength of pump
    VgsR = 5 #ratio of Vgs/Vgp
    VgiR = k2*VgsR #ratio of Vgi/Vgp
    ks0= (rindex(ls)*2*np.pi)/ls #wave-vector of signal
    ki0= k1*ks0 #wave-vector of idler
    kp0= (rindex(lp)*2*np.pi)/lp #wave-vector of pump
    kst= 0.05*ks0 #ks_tilde
    kit= k3*0.05*ki0 #ki_tilde
    return np.sinc(((2*kp0)/(np.pi*s**2))*(kst+kit-(VgsR*kst+VgiR*kit)-0.5*(((qs**2)/(ks0+kst))+((qi**2)/(ki0+kit))-(((qs+qi)**2)/(kp0+VgsR*kst+VgiR*kit)))))*np.exp(-(abs(qs + qi-(VgsR*kst+VgiR*kit))**2)/(2*p**2))

@jit(nopython=True)
def Gfunc(qs,qi): #collected JTMA involving collection function
    c = 16.
    return (30/c)*np.exp(-(qs**2+qi**2)/(2*c**2))*Ffunc(qs,qi)

G = np.identity(N) #initialisation of G matrix
for i in range (N): 
    for j in range(N):
        a_s= i-(N-1)/2
        a_i= j-(N-1)/2
        G[i][j]= Gfunc(a_s,a_i)
        print(i,j,G[i][j])
#plotting JTMA
plt.contourf(np.arange(-(N-1)/2,1+(N-1)/2),np.arange(-(N-1)/2,1+(N-1)/2),G)
plt.colorbar()
plt.plot(np.arange(-(N-1)/2,1+(N-1)/2),np.arange(-(N-1)/2,1+(N-1)/2),'w--',linewidth=0.7)
plt.plot(np.arange(-(N-1)/2,1+(N-1)/2),-np.arange(-(N-1)/2,1+(N-1)/2),'w--',linewidth=0.7)
plt.axis('square')
plt.title('JTMA   (k1=%.1f, k2=%d, k3=%d)'%(k1,k2,k3))
plt.show()

def Pr(qs, qi, s, ai):#function involving phase change 
    if qs>s and qi>ai or qs<s and qi<ai:
        return Gfunc(qs, qi)
    else:
        return -Gfunc(qs,qi)

Pr_M= np.identity(N) #initialisation of Pr matrix
for i in range(N):
    for j in range(N):
        a_i = i - (N - 1) / 2
        a_s = j - (N - 1) / 2
        I1 = sci.dblquad(Pr, -np.inf, 0, -np.inf, 0, args=[ a_s, a_i])[0]
        I2 = sci.dblquad(Pr, -np.inf, 0, 0, np.inf, args=[ a_s, a_i])[0]
        I3 = sci.dblquad(Pr, 0, np.inf, -np.inf, 0, args=[ a_s, a_i])[0]
        I4 = sci.dblquad(Pr, 0, np.inf, 0, np.inf, args=[ a_s, a_i])[0]
        Pr_M[i][j] = abs(I1+I2+I3+I4)** 2
        print(i, j, Pr_M[i][j])
#plotting Pr matrix
plt.contourf(np.arange(-(N-1)/2,1+(N-1)/2),np.arange(-(N-1)/2,1+(N-1)/2),Pr_M)
plt.colorbar()
plt.xlabel('$a_s$')
plt.ylabel('$a_i$')
plt.axis('square')
plt.title('Pr (k1=%.1f k2=%d k3=%d)'%(k1,k2,k3))
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
#importing required packages.
N= 61
#defining the experimental parameters.
Vgs= 1.61821*10**8
Vgi= 1.69947*10**8
Vgp= 1.65513*10**8

sigp= 140*10**-15
kp0= 1.42550*10**7
ki0= 7.02940*10**6
ks0= 7.36158*10**6
L=  0.002
#scaling factor.
sf= 10000
#alpha term
@jit(nopython=True)
def alpha_func(kst,kit):
    return np.exp(-0.5*(abs(3*10**8*((Vgs/Vgp)*kst+(Vgi/Vgp)*kit)*sigp)**2))
#sinc term
@jit(nopython=True)
def sinc_func(kst, kit):
    return np.sinc((L*((1-(Vgs/Vgp))*kst+(1-(Vgi/Vgp))*kit))/(2*np.pi))
#initialising alpha and sinc term
alpha = np.identity(N)
sinc= np.identity(N)
for i in range (N):
    for j in range(N):
        a_s= i-(N-1)/2
        a_i= j-(N-1)/2
        alpha[i][j]= alpha_func(a_s*sf,a_i*sf)
        sinc[i][j]= sinc_func(a_s*sf, a_i*sf)
        print(i,j,alpha[i][j])

#plotting alpha term
plt.contourf(sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), alpha)
plt.colorbar().ax.tick_params(labelsize=15)
plt.plot(sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),'w--', linewidth=0.9)
plt.plot(sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),-sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),'w--', linewidth=0.9)
plt.axis('square')
plt.xlabel('$ \widetilde{k_s}$')
plt.ylabel('$ \widetilde{k_i}$')
plt.title('alpha term', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight= 'bold')
plt.show()

#plotting sinc term
plt.contourf(sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), abs(sinc))
plt.colorbar().ax.tick_params(labelsize=15)
plt.plot(sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),'w--', linewidth=0.9)
plt.plot(sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),-sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),'w--', linewidth=0.9)
plt.axis('square')
plt.xlabel('$ \widetilde{k_s}$')
plt.ylabel('$ \widetilde{k_i}$')
plt.title('Sinc term', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight= 'bold')
plt.show()

#plotting JSA
plt.contourf(sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), abs(alpha*sinc))
plt.colorbar().ax.tick_params(labelsize=15)
plt.plot(sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),'w--', linewidth=0.9)
plt.plot(sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),-sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),'w--', linewidth=0.9)
plt.axis('square')
plt.xlabel('$ \widetilde{k_s}$')
plt.ylabel('$ \widetilde{k_i}$')
plt.title('Joint Spectral Amplitude (JSA)', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight= 'bold')
plt.show()

#plotting pump spectral bandwidth
column= int(0.5*(N-1))
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.spines['left'].set_position('center')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
plt.plot(sf*np.arange(-(N - 1) / 2, 1 + (N - 1) / 2),alpha[:][column],'o-')
plt.xticks(fontweight='bold')
plt.yticks(fontweight= 'bold')
plt.show()

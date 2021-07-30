import scipy.integrate as sci
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
#importing modules
N= 53 #int(input('N= ')) the dimension of Pr matrix is NxN

@jit(nopython=True)
def G(qs,qi,e):
    p = 2
    c = 16.
    s = 2.5 * c
    ci = abs(1 / (1 - e)) #cs=1 and ci is varied as per e.
    return (30./c)*np.exp(- ((p**2+c**2)*(qs**2+qi**2)+2*qs*qi*c**2)/(2*p**2*c**2))* np.sinc(((2+e)*0.5*abs(qs-ci*qi)**2 + e*0.5*abs(qs+ci*qi)**2)/(np.pi*s**2)) #simplified collected JTMA function.

for e in np.arange(0,1,0.1): #loop for different values of e
    Pr1 = np.identity(N) #initialising collected JTMA
    for i in range (N):# loop for different a_s value
        for j in range(N):# loop for different a_i value
            a_s= i-(N-1)/2
            a_i= j-(N-1)/2
            Pr1[i][j]= G(a_s,a_i,e)

    plt.contourf(np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), Pr1)
    plt.colorbar()
    plt.xlabel('$a_s$')
    plt.ylabel('$a_i$')
    plt.title('JTMA $\epsilon$ = %.2f'%e)
    plt.show() #visualising collected JTMA

    Pr=np.identity(N) #initialising Pr matrix 
    for i in range (N):# loop for different a_s value
        for j in range (N):# loop for different a_i value
            a_i= i-(N-1)/2
            a_s= j-(N-1)/2
            I1= sci.dblquad(G,-np.inf,a_s,-np.inf, a_i, args=[e])[0]#G++
            I2= sci.dblquad(G, a_s, np.inf,a_i, np.inf, args=[e])[0]#G--
            I3= sci.dblquad(G,-np.inf, a_s,a_i, np.inf, args=[e])[0]#G-+
            I4= sci.dblquad(G,a_s, np.inf, -np.inf,a_i, args=[e])[0]#G+-
            Pr[i][j] = abs(I1 + I2 - I3 - I4)**2
            print(i,j,Pr[i][j])

    plt.contourf(np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), Pr)
    plt.colorbar()
    plt.xlabel('$a_s$')
    plt.ylabel('$a_i$')
    plt.title('Pr($a_s$, $a_i$) $\epsilon$ = %.2f' % e)
    plt.show() #visualising Pr matrix.

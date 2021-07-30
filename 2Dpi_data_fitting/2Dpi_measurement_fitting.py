import time

begin= time.time()

from numba import jit
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import special

file_name= '2dpi.mat'#input('File Name: ')
pi_data= sio.loadmat(file_name)
A= pi_data['JXACW'] #numpy array containing the data
sf= 4.34

@jit(nopython=True)
def pos(array):
    sum1= np.sum(array, axis=1) #summing column wise
    sum2= np.sum(array, axis=0) #summing row wise
    i, = np.where(sum1 == np.min(sum1)) #finding the row index where the sum1 is minimum
    j, = np.where(sum2 == np.min(sum2)) #finding the column index where the sum2 is minimum
    return i[0], j[0]

X=pos(A)[0]
Y=pos(A)[1] #position of the origin (X,Y)

def Pr(array):
    Pr1= np.diagonal(array,offset=abs(X-Y))
    Pr2= np.diagonal(np.fliplr(array), offset= (array.shape[0] -X-Y-1))
    return Pr1, Pr2 #slicing the data along the diagonal to find Pr(a,a) and Pr(a,-a)

x1= np.arange(Pr(A)[0].shape[0])-X
y1= Pr(A)[0] #Experimental Pr(a,-a)

def func1(x,N1,N2, sigma_c):
    return abs(N1- N2*np.exp(-(x**2)/(sigma_c**2)))**2

popt1, pcov1 = curve_fit(func1, x1, y1, absolute_sigma=True)
err1= np.sqrt(np.diag(pcov1))
c= abs(popt1[2])

x2= np.arange(Pr(A)[1].shape[0])-X
y2= Pr(A)[1] #Experimental Pr(a,a)

def func2(x,C1,C2,C3,C4,C5,p):
    return np.exp(-C1*(x**2)/(2*c**2))*(C2*np.sqrt(2)*p*np.exp(-2*(x**2)*(1./(c**2) +
    1./(p**2))) - C3*2*np.sqrt(np.pi)*abs(x)*np.exp(-x**2/c**2)*special.erfc(np.sqrt(2)*abs(x)
    *(1/(2*c**2)+ 1/p**2)**0.5) - C3*c*np.pi*0.5*(1-2*special.erf(abs(x)/c))+C4)**2 +C5

popt2, pcov2 = curve_fit(func2, x2, y2,absolute_sigma=True)
err2= np.sqrt(np.diag(pcov2))

print('sigma_p = %f rad/mm error = %f rad/mm' %(abs(popt2[5])*sf,err2[5]*sf))
print('sigma_c = %f rad/mm error = %f rad/mm'%(abs(popt1[2])*sf,err1[2]*sf))

end= time.time()
print('Runtime of the program = %f'%(end-begin))

plt.title('$\lambda_{s}, \lambda_{i} = 810 nm$')
plt.plot(7*x1,y1,'o', label= 'Experimental')
plt.plot(7*x1, func1(x1,*popt1),'r--',
    label= 'fit: $\sigma_c=%.3f \pm %f$ rad/mm'%(abs(popt1[2])*sf,err1[2]*sf))
plt.xlabel('a (pixel)')
plt.ylabel('$C_0$ Pr(a,-a)')
plt.legend(loc=1)
plt.show()

plt.title('$\lambda_{s}, \lambda_{i} = 810 nm$')
plt.plot(7*x2,y2,'o',label= 'Experimental')
plt.plot(7*x2, func2(x2,*popt2),'r--',
    label= 'fit: $\sigma_p=%.3f \pm %.3f$ rad/mm'%(abs(popt2[5])*sf,err2[5]*sf))
plt.xlabel('a (pixel)')
plt.ylabel('$C_0$ Pr(a,a)')
plt.legend(loc=1)
plt.show()
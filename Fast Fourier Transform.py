from scipy.fft import fft
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
%matplotlib notebook

"""User parameters"""

N = 512               # Number of sample points to be taken
start = 3897         # starting index corresponding to the region of interest in the dataset
stop = 24063         # ending index corresponding to the region of interest in the dataset
dat = np.loadtxt(r"C:\Users\mails\Desktop\Et sidus oritur\datasets\model32.dat")[start:stop:,3] #original spliced dataset
t = (np.loadtxt(r"C:\Users\mails\Desktop\Et sidus oritur\datasets\model32.dat")[start:stop:,0]) #original spliced timedata
T =  (np.amax(t)-np.amin(t))/N              #sample spacing also known as periodicity
kernel = 100          # time window for the moving average

"""Computation"""

def movingAverage(x,y,N=100):  #function calculating and storing moving average 
    arrSize = len(x)
    m=np.array(np.zeros(arrSize))
    for i in range(N):
        np.put(m,i, (np.sum((y[0:i+N]))/(i+N)))
    for i in range(N,arrSize-N):
        np.put(m,i,(np.sum(y[i-N:i+N+1])/(2*N+1)))
    for i in range(arrSize-N,arrSize):
        np.put(m,i, (np.sum((y[i-N:arrSize]))/(arrSize-i+N)))
    return(m)
m = movingAverage(t,dat,kernel) # function call with the parameters being the spliced datasets and the kernel that was set
y = dat - m                     # removing the running average from the data to yield a dataset that oscillates about 0
Y = fft(y)                      # taking the fourier transform of the filtered data
yf = (Y*np.conj(Y))*1000        # Computing Power spectral Density, instead of using a simple fft
xf = np.linspace(0.0, 1.0/(2.0*T), N//2) # Generating a frequency scale to use in the plot

"""Plotting the results"""

import matplotlib.pyplot as plt
fig1=plt.figure(dpi=108,figsize=(7,7))
ax1= fig1.add_subplot(111)a
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#plt.xlim(0.0002,0.0006)        # used to read the specific peaks better, can comment out when not needed
plt.ylabel("Amplitude", fontsize = 16)
plt.xlabel("Frequency",fontsize = 16)
plt.title('Power Spectral Density', fontsize =18)
plt.grid()
plt.tight_layout()
plt.savefig('finalfinalspectrum.jpg')
plt.show()
print(T)
f = np.where((2.0/N * np.abs(yf[0:N//2]))==np.amax(2.0/N * np.abs(yf[0:N//2])))
F = xf[f]
P = (1/(2*F))
print( F,T)
print("The period corresponding to highest peak: ",P, " years")
print((np.amax(t)-np.amin(t)))

import numpy as  np
import matplotlib.pyplot as plt
%matplotlib notebook
import math
"""User parameters"""
untchy = np.loadtxt(r"C:\Users\mails\Desktop\Et sidus oritur\datasets\model32.dat")[:,3]
y = (np.log(np.loadtxt(r"C:\Users\mails\Desktop\Et sidus oritur\datasets\model32.dat")[:,3]))  # log of original complete dataset
nt = (np.loadtxt(r"C:\Users\mails\Desktop\Et sidus oritur\datasets\model32.dat")[:,0])                                                           #corresponding even time data
x = (np.log(nt))                                                                             # log of even time data
kernel = 100                                                  # time window for the moving average

"""Computation"""

def movingAverage(x,y,N=1000):                                # function calculating and storing moving average 
    arrSize = len(x)
    m=np.array(np.zeros(arrSize))
    for i in range(N):
        np.put(m,i, (np.sum((y[0:i+N]))/(i+N)))
    for i in range(N,arrSize-N):
        np.put(m,i,(np.sum(y[i-N:i+N+1])/(2*N+1)))
    for i in range(arrSize-N,arrSize):
        np.put(m,i, (np.sum((y[i-N:arrSize]))/(arrSize-i+N)))
    return(m)
m = movingAverage(x,y,kernel)         # function call with the parameters being the spliced datasets and the kernel that was set
filt = y-m                            # removing the running average from the data to yield a dataset that oscillates about 0

"""Plotting the results"""

line,=plt.plot(x/2.3,y/2.3)                   # original unfiltered data being plotted
avg,=plt.plot(x/2.3,m/2.3,label='Moving average')
filtered,=plt.plot(x/2.3,filt/2.3)            # filtered data
plt.xlim(4.4,6)                  # toggle to magnify and check the region of interest
plt.title("Moving average-burst region",fontsize = 14)
plt.legend([line,avg,filtered],['Mass accretion vs time','Moving average','filtered'])
plt.ylabel(r'$\log(\dot{M})$  $[(M_\odot)\cdot(yr^{-1}$]', fontsize = 12)
plt.xlabel(r'$\log(time)$ [yr]',fontsize = 12)
idx=np.where(x>=10)                 # toggle
edx=np.where(x<=13.7)                # toggle
print("start: ", np.amin(idx))        # yields starting index corresponding to the region of interest in the dataset
print("stop: ", np.amax(edx))  
print()                                # yields ending index corresponding to the region of interest in the dataset
plt.grid()
plt.tight_layout()
plt.savefig('finalfinalwindow.pdf')
print(len(y))
print(len(untchy))
print(np.amax(nt))

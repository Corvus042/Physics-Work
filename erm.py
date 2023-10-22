
import pandas as pd
import csv
import cv2
import sys
import numpy as np
import plotext
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.backends.backend_pdf import PdfPages

pix = 13.21

#excel spreadsheet of data
data = pd.read_excel("C:/Users/Nwgol/Onedrive/Desktop/March 2023 Tracking/Jan 5th/data/Jan 5th C5_e xy.xlsm")
x = list(data.x*(1/pix))
y = list(data.y*(1/pix))



#Ploting MSD vs Time
MSDxtplot = np.zeros(len(x)-1)
MSDytplot = np.zeros(len(x)-1)
MSDtotaltplot = np.zeros(len(x)-1)


for n in range(len(x)-1):
    dx = 0
    dy = 0
    total = 0
    m = n+1
    while m <(len(x)):
        dx += (x[m]-x[m-n])**2
        dy += (y[m]-y[m-n])**2
        total = dx+dy
        m +=1 
    MSDxtplot[n] = ((dx)/((len(x))-n))
    MSDytplot[n] = ((dy)/((len(x))-n))
    MSDtotaltplot[n] = total/(len(x)-n)

#Plotting MSD vs lag time
MSDylag = []
MSDxlag =[]
MSDtotallag = []
k = 1
while k < len(x):
    dly = []
    dlx = []
    dltotal = []
    for j in range(len(x)-k):
        dly.append((y[j+k]-y[j])**2)
        dlx.append((x[j+k]-x[j])**2)
        dltotal.append(((x[j+k]-x[j])**2)+((y[j+k]-y[j])**2))

    MSDxlag.append(np.mean(dlx))
    MSDylag.append(np.mean(dly))
    MSDtotallag.append(np.mean(dltotal))
    k+=1


lag = np.linspace(0,len(x)/15, len(MSDxlag))


time = len(x)/15
t = np.linspace(0,time,len(MSDytplot))

#Plotting 
fig1, axis = plt.subplots(1)
axis.scatter(t,MSDytplot)
axis.set_xscale('log')
axis.set_yscale('log')
axis.set_title("y-axis MSD with respect to time")

fig14, axis =plt.subplots(1)
axis.scatter(t,MSDtotaltplot)
axis.set_xscale('log')
axis.set_yscale('log')
axis.set_title("total MSD with respect to time")



fig12, axis = plt.subplots(1)
axis.scatter(lag,MSDylag)
axis.set_title("Y-axis Translational MSD vs Lag Time ")
fig13, axis = plt.subplots(1)
axis.scatter(lag,MSDxlag)
axis.set_title("X-axis Translational MSD vs Lag Time ")
fig11, axis = plt.subplots(1)
axis.scatter(lag,MSDtotallag)
axis.set_title("Translational MSD vs Lag Time ")


  


p = PdfPages("C:/Users/Nwgol/Onedrive/Desktop/March 2023 Tracking/Jan 5th/data/Jan 5th C5_e graphs.pdf")
      

fig1.savefig(p, format='pdf')
fig14.savefig(p, format = 'pdf')

fig12.savefig(p,format = 'pdf')
fig13.savefig(p,format = 'pdf')
fig11.savefig(p,format = 'pdf')


#path = "C:/Users/Nwgol/Onedrive/Desktop/March 2023 Tracking/Dataxy/Graph Images/34.0C/"
#fig1.savefig(path+'MSD vs Time.png', dpi=fig1.dpi)
#fig7.savefig(path+'Histogram of Delta Theta.png', dpi=fig7.dpi)
#fig9.savefig(path+'Delta Theta vs Time.png', dpi=fig9.dpi)
#fig3.savefig(path+'MSD.png', dpi=fig3.dpi)
#fig4.savefig(path+'Diffusion.png', dpi=fig4.dpi)


p.close()  

#frames = np.linspace(1,len(x)+1,len(x))

dct2 = {  
      'Time in seconds': t, 
      'y-axis MSD in um^2': MSDytplot ,
      'Total MSD in um^2': MSDtotaltplot
    } 
dct3 = {
      'Lag Time in seconds': lag,
      'y-axis MSD in um^2': MSDylag,
      'x-axis MSD in um^2': MSDxlag,
      'Total MSD in um^2': MSDtotallag
}


data2 = pd.DataFrame(dct2)
data3 = pd.DataFrame(dct3)
  
writer = pd.ExcelWriter("C:/Users/Nwgol/Onedrive/Desktop/March 2023 Tracking/Jan 5th/data/Jan 5th C5_e Data.xlsx",  engine='xlsxwriter')
data2.to_excel(writer, sheet_name = 'data set 1')
data3.to_excel(writer, sheet_name = 'data set 2')
writer.close()
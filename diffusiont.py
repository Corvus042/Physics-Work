
import pandas as pd
import csv
import cv2
import sys
import numpy as np
import plotext
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.backends.backend_pdf import PdfPages

pix = 17.39

#excel spreadsheet of data
data = pd.read_excel("C:/Users/Nwgol/Onedrive/Desktop/March 2023 Tracking/Data/RT S3 xy.xlsm")
x = list(data.x*(1/pix))
y = list(data.y*(1/pix))
phi = list(data.phi)
#phi = x

#For specific time frames 
#x = x[900:1900]
#y = y[900:1900]
#phi = phi[900:1900]

#If angle needs correction 
#phi = list(data.phi-90)
#phi = ((np.sign(phi)-1)/2)*(-180)+(phi)



#Angle list for rotational matric
theta = np.linspace(0,2*np.pi,101)

#Collection lists
MSDyplot =[]
MSDxplot =[]
diffusionxlist = []
diffusionylist = []

#For MSD and rotational matirx
for l in range(len(theta)): 
    xtheta = [] #x and y collectors for each angle l between 0 and 2pi
    ytheta = []
    for k in range(len(x)): #applying the rotaional matrix to each point 
        xtheta.append(x[k]*np.cos(theta[l])-y[k]*np.sin(theta[l]))
        ytheta.append(x[k]*np.sin(theta[l])+y[k]*np.cos(theta[l]))

    i = 1
    MSDxlist = []
    MSDylist = []
    MSDtotallist = []
    while i<21:
        xj = []
        yj = []
        for j in range(0,len(xtheta),i): #statistical difference 
            xj.append(xtheta[j]) 
            yj.append(ytheta[j])

        #calculation of MSD 
        dx = np.diff(xj)
        dy = np.diff(yj)
        xsq= [each_x**2 for each_x in dx]
        ysq= [each_y**2 for each_y in dy]
        MSDx = np.sum(xsq)/len(x)
        MSDy = np.sum(ysq)/len(y)
        MSDxlist.append(MSDx)
        MSDylist.append(MSDy)
        i +=1
    MSDxav = np.average(MSDxlist)
    MSDyav = np.average(MSDylist)
    MSDyplot.append(MSDyav)
    MSDxplot.append(MSDxav)
    tou = 0.0333
    time = len(x)/15 #requires that all frames are tracked 
    diffusionx = (MSDxav*.5)/(tou)
    diffusiony = (MSDyav*.5)/(tou)
    diffusionxlist.append(diffusionx)
    diffusionylist.append(diffusiony)


#Ploting MSD vs Time
MSDxtplot = np.zeros(len(x)-1)
MSDytplot = np.zeros(len(x)-1)
MSDtotaltplot = np.zeros(len(x)-1)
MSDphiplot = np.zeros(len(x)-1)

for n in range(len(x)-1):
    dx = 0
    dy = 0
    total = 0
    dphi = 0
    m = n+1
    while m <(len(x)):
        dx += (x[m]-x[m-n])**2
        dy += (y[m]-y[m-n])**2
        total = dx+dy
        dphi+= (phi[m]-phi[m-n])
        m +=1 
    MSDxtplot[n] = ((dx)/((len(x))-n))
    MSDytplot[n] = ((dy)/((len(x))-n))
    MSDphiplot[n] = ((dphi)/((len(x))-n))
    MSDtotaltplot[n] = total/(len(x)-n)

#Calculation of angular diffusion
dphi = np.diff(phi)
phisq = [each_phi**2 for each_phi in dphi]
MSDphi = np.mean(phisq)
diffphi = (MSDphi*.5)/tou


#Plotting MSD vs lag time
MSDphilag = []
MSDylag = []
MSDxlag =[]
MSDtotallag = []
k = 1
while k < len(phi):
    dlphi = []
    dly = []
    dlx = []
    dltotal = []
    for j in range(len(phi)-k):
        dlphi.append((phi[j+k]-phi[j])**2)
        dly.append((y[j+k]-y[j])**2)
        dlx.append((x[j+k]-x[j])**2)
        dltotal.append(((x[j+k]-x[j])**2)+((y[j+k]-y[j])**2))

    MSDxlag.append(np.mean(dlx))
    MSDphilag.append(np.mean(dlphi))
    MSDylag.append(np.mean(dly))
    MSDtotallag.append(np.mean(dltotal))
    k+=1


lag = np.linspace(0,len(phi)/15, len(MSDphilag))



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

#fig2, axis2 = plt.subplots(1)
#axis2.plot((t), (MSDxtplot))
#axis2.set_xscale('log')
#axis2.set_yscale('log')
#axis2.set_title("x-axis MSD with respect to time")

#fig6, axis = plt.subplots(1)
#axis.plot(t,MSDphiplot)
#axis.set_xscale('log')
#axis.set_yscale('log')
#axis.set_title("Angular MSD with respect to time")

fig7, axis = plt.subplots(1)
axis.hist(dphi, bins = 50 )
axis.set_title("Histogram of Delta Theta")
fig8, axis = plt.subplots(1)
axis.scatter(lag,MSDphilag)
axis.set_title("Angular MSD vs Lag Time ")
fig12, axis = plt.subplots(1)
axis.scatter(lag,MSDylag)
axis.set_title("Y-axis Translational MSD vs Lag Time ")
fig13, axis = plt.subplots(1)
axis.scatter(lag,MSDxlag)
axis.set_title("X-axis Translational MSD vs Lag Time ")
fig11, axis = plt.subplots(1)
axis.scatter(lag,MSDtotallag)
axis.set_title("Translational MSD vs Lag Time ")
fig9, axis = plt.subplots(1)
axis.plot(t,dphi)
axis.set_title("Delta Theta with respect to time")

#tl = list(np.log10(t))
#MSDytplotl = list(np.log10(MSDytplot))
#MSDxtplotl = list(np.log10(MSDxtplot))
#diffx = linregress(t[1:400],MSDxtplot[1:400])
#diffy = linregress(t[1:400], MSDylag[1:400])
#print(diffx)
#print(diffy)
  
#ma = MSDyplot.index(max(MSDyplot))
#MSDyplot = MSDyplot[-(len(MSDyplot)-ma):] + MSDyplot[:-(len(MSDyplot)-ma)]
#ma2 = diffusionylist.index(max(diffusionylist))
#diffusionylist = diffusionylist[-(len(diffusionylist)-ma2):] + diffusionylist[:-(len(diffusionylist)-ma2)]

#xma = MSDxplot.index(min(MSDxplot))
#MSDxplot = MSDxplot[-(len(MSDxplot)-xma):] + MSDxplot[:-(len(MSDxplot)-xma)]
#xma2 = diffusionxlist.index(min(diffusionxlist))
#diffusionxlist = diffusionxlist[-(len(diffusionxlist)-xma2):] + diffusionxlist[:-(len(diffusionxlist)-xma2)]

fig3, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
ax1.set_theta_direction(1)
ax1.set_theta_offset(np.pi / 2.0)
ax1.plot(theta, MSDxplot)
ax1.set_title("x-axis MSD")
ax2.set_theta_direction(1)
ax2.set_theta_offset(np.pi / 2.0)
ax2.plot(theta, MSDyplot)
ax2.set_title("y-axis MSD")

fig4, (ax3, ax4) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
ax3.set_theta_direction(1)
ax3.set_theta_offset(np.pi / 2.0)
ax3.plot(theta, diffusionxlist)
ax3.set_title("x-axis Diffusion")
ax4.set_theta_direction(1)
ax4.set_theta_offset(np.pi / 2.0)
ax4.plot(theta, diffusionylist)
ax4.set_title("y-axis Diffusion")

d1 = list(theta[0:100:25]*(180/(np.pi)))
d3 = list(diffusionylist[0:100:25])
d1.append('Angular Diffusion')
d3.append(diffphi)

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
fig5, axs = plt.subplots(1, 1)
#data = np.dstack((theta[0:100:11],diffusionxlist[0:100:11],diffusionylist[0:100:11]))
data = np.dstack(np.dstack((d1,d3)))
columns = ("Theta in Degrees", "Y-axis Diffusion in um^2/S")
axs.axis('tight')
axs.axis('off')
the_table = axs.table(cellText=data, colLabels=columns, loc='center')


p = PdfPages("C:/Users/Nwgol/Onedrive/Desktop/March 2023 Tracking/Data/RT S3 graphs.pdf")
      

fig1.savefig(p, format='pdf')
fig14.savefig(p, format = 'pdf')
#fig2.savefig(p, format='pdf')
#fig6.savefig(p,format = 'pdf')
fig7.savefig(p,format = 'pdf')
fig8.savefig(p,format = 'pdf')
fig12.savefig(p,format = 'pdf')
fig13.savefig(p,format = 'pdf')
fig11.savefig(p,format = 'pdf')
#fig9.savefig(p,format = 'pdf')
fig3.set_size_inches(10,5)
fig3.savefig(p, format='pdf')
fig4.set_size_inches(10,5)
fig4.savefig(p, format='pdf')
fig5.savefig(p,format = 'pdf')

#path = "C:/Users/Nwgol/Onedrive/Desktop/March 2023 Tracking/Dataxy/Graph Images/34.0C/"
#fig1.savefig(path+'MSD vs Time.png', dpi=fig1.dpi)
#fig7.savefig(path+'Histogram of Delta Theta.png', dpi=fig7.dpi)
#fig9.savefig(path+'Delta Theta vs Time.png', dpi=fig9.dpi)
#fig3.savefig(path+'MSD.png', dpi=fig3.dpi)
#fig4.savefig(path+'Diffusion.png', dpi=fig4.dpi)


p.close()  

#frames = np.linspace(1,len(x)+1,len(x))
dct = {'Theta in deg': theta*(180/(np.pi)), 
      'x MSD in um^2': MSDxplot,
      'y MSD in um^2': MSDyplot,
      'x Diffusion in um^2/s': diffusionxlist,
      'y Diffusion in um^2/S': diffusionylist,  
    } 
  
dct2 = {  
      'Time in seconds': t, 
      'y-axis MSD in um^2': MSDytplot ,
      'Total MSD in um^2': MSDtotaltplot
    } 
dct3 = {
      'Lag Time in seconds': lag,
      'Angular MSD in deg^2': MSDphilag,
      'y-axis MSD in um^2': MSDylag,
      'x-axis MSD in um^2': MSDxlag,
      'Total MSD in um^2': MSDtotallag
}


data = pd.DataFrame(dct) 
data2 = pd.DataFrame(dct2)
data3 = pd.DataFrame(dct3)
  
writer = pd.ExcelWriter("C:/Users/Nwgol/Onedrive/Desktop/March 2023 Tracking/Data/RT S3 Data.xlsx",  engine='xlsxwriter')
data.to_excel(writer, sheet_name = 'data set 1')
data2.to_excel(writer, sheet_name = 'data set 2')
data3.to_excel(writer, sheet_name = 'data set 3')
writer.close()
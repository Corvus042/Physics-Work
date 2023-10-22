
import pandas as pd
import cv2
import sys
import numpy as np
import plotext

data = pd.read_excel("C:/Users/Nwgol/Onedrive/Desktop/xypart4.xlsm")

x = list(data.x)
y = list(data.y)
x_values = []
y_values = []

for i in range(len(x)):
    if x[i] == 0:
        pass
    else:
        x_values.append(x[i])
        y_values.append(y[i])

theta = [0]
for i in range(9):
    next = theta[i]+(((np.pi)/2)/9) 
    theta.append(next)

print(theta)

x_values1 = []
x_values2 = []
x_values3 = []
x_values4 = []
x_values5 = []
x_values6 = []
x_values7 = []
x_values8 = []
x_values9 = []
x_values10 = []

y_values1 = []
y_values2 = []
y_values3 = []
y_values4 = []
y_values5 = []
y_values6 = []
y_values7 = []
y_values8 = []
y_values9 = []
y_values10 = []




for i in range(len(x_values)):
    x_values1.append(x_values[i]*np.cos(theta[0])+y_values[i]*np.sin(theta[0]))
    x_values2.append(x_values[i]*np.cos(theta[1])+y_values[i]*np.sin(theta[1]))
    x_values3.append(x_values[i]*np.cos(theta[2])+y_values[i]*np.sin(theta[2]))
    x_values4.append(x_values[i]*np.cos(theta[3])+y_values[i]*np.sin(theta[3]))
    x_values5.append(x_values[i]*np.cos(theta[4])+y_values[i]*np.sin(theta[4]))
    x_values6.append(x_values[i]*np.cos(theta[5])+y_values[i]*np.sin(theta[5]))
    x_values7.append(x_values[i]*np.cos(theta[6])+y_values[i]*np.sin(theta[6]))
    x_values8.append(x_values[i]*np.cos(theta[7])+y_values[i]*np.sin(theta[7]))
    x_values9.append(x_values[i]*np.cos(theta[8])+y_values[i]*np.sin(theta[8]))
    x_values10.append(x_values[i]*np.cos(theta[9])+y_values[i]*np.sin(theta[9]))

    y_values1.append(-x_values[i]*np.sin(theta[0])+y_values[i]*np.cos(theta[0]))
    y_values2.append(-x_values[i]*np.sin(theta[1])+y_values[i]*np.cos(theta[1]))
    y_values3.append(-x_values[i]*np.sin(theta[2])+y_values[i]*np.cos(theta[2]))
    y_values4.append(-x_values[i]*np.sin(theta[3])+y_values[i]*np.cos(theta[3]))
    y_values5.append(-x_values[i]*np.sin(theta[4])+y_values[i]*np.cos(theta[4]))
    y_values6.append(-x_values[i]*np.sin(theta[5])+y_values[i]*np.cos(theta[5]))
    y_values7.append(-x_values[i]*np.sin(theta[6])+y_values[i]*np.cos(theta[6]))
    y_values8.append(-x_values[i]*np.sin(theta[7])+y_values[i]*np.cos(theta[7]))
    y_values9.append(-x_values[i]*np.sin(theta[8])+y_values[i]*np.cos(theta[8]))
    y_values10.append(-x_values[i]*np.sin(theta[9])+y_values[i]*np.cos(theta[9]))


dif = []

while i < 9:
    MSDaccum = 0

i = 1
MSDlist = []
while i<11:
    xj = []
    yj = []
    for j in range(0,len(x_values),i):
        xj.append(x_values[j])
        yj.append(y_values[j])
    xsq= [each_x**2 for each_x in xj]
    ysq= [each_y**2 for each_y in yj]
    r = np.sqrt(xsq+ysq)
    diff = np.diff(r) #this calculates r(t + dt) - r(t)
    diff_sq = diff**2
    MSD = np.mean(diff_sq)
    MSDlist.append(MSD)
    i +=1    
MSD = np.average(MSDlist)
time = len(x_values)/30
diffusion = (MSD*.5)/(time)
print(diffusion)

xsq= [each_x**2 for each_x in x_values]
ysq= [each_y**2 for each_y in y_values1]
r = np.sqrt(xsq+ysq)
diff = np.diff(r) #this calculates r(t + dt) - r(t)
diff_sq = diff**2
MSD = np.mean(diff_sq)
print(MSD)
time = len(x_values)/30
diffusion1 = (MSD*.5)/(time)
print(diffusion1)
dif.append(diffusion1)

xsq= [each_x**2 for each_x in x_values2]
ysq= [each_y**2 for each_y in y_values2]
r = np.sqrt(xsq+ysq)
diff = np.diff(r) #this calculates r(t + dt) - r(t)
diff_sq = diff**2
MSD = np.mean(diff_sq)
print(MSD)
time = len(x_values)/30
diffusion2 = (MSD*.5)/(time)
dif.append(diffusion2)

xsq= [each_x**2 for each_x in x_values3]
ysq= [each_y**2 for each_y in y_values3]
r = np.sqrt(xsq+ysq)
diff = np.diff(r) #this calculates r(t + dt) - r(t)
diff_sq = diff**2
MSD = np.mean(diff_sq)
print(MSD)
time = len(x_values)/30
diffusion3 = (MSD*.5)/(time)
dif.append(diffusion3)

xsq= [each_x**2 for each_x in x_values4]
ysq= [each_y**2 for each_y in y_values4]
r = np.sqrt(xsq+ysq)
diff = np.diff(r) #this calculates r(t + dt) - r(t)
diff_sq = diff**2
MSD = np.mean(diff_sq)
print(MSD)
time = len(x_values)/30
diffusion4 = (MSD*.5)/(time)
dif.append(diffusion4)

xsq= [each_x**2 for each_x in x_values5]
ysq= [each_y**2 for each_y in y_values5]
r = np.sqrt(xsq+ysq)
diff = np.diff(r) #this calculates r(t + dt) - r(t)
diff_sq = diff**2
MSD = np.mean(diff_sq)
print(MSD)
time = len(x_values)/30
diffusion5 = (MSD*.5)/(time)
dif.append(diffusion5)

xsq= [each_x**2 for each_x in x_values6]
ysq= [each_y**2 for each_y in y_values6]
r = np.sqrt(xsq+ysq)
diff = np.diff(r) #this calculates r(t + dt) - r(t)
diff_sq = diff**2
MSD = np.mean(diff_sq)
print(MSD)
time = len(x_values)/30
diffusion6 = (MSD*.5)/(time)
dif.append(diffusion6)

xsq= [each_x**2 for each_x in x_values7]
ysq= [each_y**2 for each_y in y_values7]
r = np.sqrt(xsq+ysq)
diff = np.diff(r) #this calculates r(t + dt) - r(t)
diff_sq = diff**2
MSD = np.mean(diff_sq)
print(MSD)
time = len(x_values)/30
diffusion7 = (MSD*.5)/(time)
dif.append(diffusion7)

xsq= [each_x**2 for each_x in x_values8]
ysq= [each_y**2 for each_y in y_values8]
r = np.sqrt(xsq+ysq)
diff = np.diff(r) #this calculates r(t + dt) - r(t)
diff_sq = diff**2
MSD = np.mean(diff_sq)
print(MSD)
time = len(x_values)/30
diffusion8 = (MSD*.5)/(time)
dif.append(diffusion8)

xsq= [each_x**2 for each_x in x_values9]
ysq= [each_y**2 for each_y in y_values9]
r = np.sqrt(xsq+ysq)
diff = np.diff(r) #this calculates r(t + dt) - r(t)
diff_sq = diff**2
MSD = np.mean(diff_sq)
print(MSD)
time = len(x_values)/30
diffusion9 = (MSD*.5)/(time)
dif.append(diffusion9)

xsq= [each_x**2 for each_x in x_values10]
ysq= [each_y**2 for each_y in y_values10]
r = np.sqrt(xsq+ysq)
diff = np.diff(r) #this calculates r(t + dt) - r(t)
diff_sq = diff**2
MSD = np.mean(diff_sq)
print(MSD)
time = len(x_values)/30
diffusion10 = (MSD*.5)/(time)
dif.append(diffusion10)

print(dif)
plotext.scatter(theta,dif)
plotext.show()
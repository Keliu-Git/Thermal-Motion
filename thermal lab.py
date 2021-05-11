import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x_pos1, y_pos1 = np.loadtxt('data4 tracker.txt', skiprows = 2, unpack = True)
x_pos2, y_pos2 = np.loadtxt('data6 tracker.txt', skiprows = 2, unpack = True)
x_pos3, y_pos3 = np.loadtxt('Data7 tracker.txt', skiprows = 2, unpack = True)
x_pos4, y_pos4 = np.loadtxt('data9 tracker.txt', skiprows = 2, unpack = True)


def rayleigh (r, a):
    return r/(a)*np.exp(-r**2/(2*a))

def boltzmann (x):
    '''
    k = 3pi*r*η*(2Dt)/(T*t)
    '''
    x = x*10**-12 #1 micrometer^2 = 1*10^-12 meter^2
    k = 3*np.pi*(9.5*10**-7)*0.001*x/(296.5*0.5) #1g/cm-s = 0.001 kg/m-s; 0.95 micrometer = 9.5*10^-7 meter
    return k

#Calculate uncertainty for k
def boltzmann_err(x, u_x):
    '''
    u_r = 0.05 micrometer
    u_n = 0.05 centipoise
    u_T = 0.5K
    u_t = 0.03s
    '''
    x = x*10**-12
    u_x = u_x*10**-12
    u_r = 0.05*10**-7
    u_n = 0.05/1000
    u_T = 0.5
    u_t = 0.03
    rn = (9.5*10**-7)*0.001
    Tt = 296.5*0.5
    u_rn = rn*np.sqrt((u_r/9.5*10**-7)**2+(u_n/0.001)**2)
    u_Tt = Tt*np.sqrt((u_T/296.5)**2+(u_t/0.5)**2)
    rnTt = rn/Tt
    u_rnTt = rnTt*np.sqrt((u_rn/rn)**2+(u_Tt/Tt)**2)
    a = rnTt*x
    u_a = a*np.sqrt((u_x/x)**2+(u_rnTt/rnTt)**2)
    return 3*np.pi*u_a

#Uncertainty of NA
def avo_err(k, u_k):
    u_a = 1/k*(u_k/k)
    return 8.3145*u_a
    
def avogadro(k):
    return 8.3145/k

dx1 = np.zeros(len(x_pos1)-1)
dy1 = np.zeros(len(y_pos1)-1)

dr_sq1 = np.zeros(len(dx1))

dx2 = np.zeros(len(x_pos2)-1)
dy2 = np.zeros(len(y_pos2)-1)

dr_sq2 = np.zeros(len(dx2))

dx3 = np.zeros(len(x_pos3)-1)
dy3 = np.zeros(len(y_pos3)-1)

dr_sq3 = np.zeros(len(dx3))

dx4 = np.zeros(len(x_pos4)-1)
dy4 = np.zeros(len(y_pos4)-1)

dr_sq4 = np.zeros(len(dx4))


#0.1155 micrometer/pixel
for i in range (0, len(x_pos1)-1):
    dx1[i] = (x_pos1[i+1]-x_pos1[i])*0.1155
    dy1[i] = (y_pos1[i+1]-y_pos1[i])*0.1155
    dr_sq1[i] = dx1[i]**2+dy1[i]**2

for i in range (0, len(x_pos2)-1):
    dx2[i] = (x_pos2[i+1]-x_pos2[i])*0.1155
    dy2[i] = (y_pos2[i+1]-y_pos2[i])*0.1155
    dr_sq2[i] = dx2[i]**2+dy2[i]**2
    
for i in range (0, len(x_pos3)-1):
    dx3[i] = (x_pos3[i+1]-x_pos3[i])*0.1155
    dy3[i] = (y_pos3[i+1]-y_pos3[i])*0.1155
    dr_sq3[i] = dx3[i]**2+dy3[i]**2

for i in range (0, len(x_pos4)-1):
    dx4[i] = (x_pos4[i+1]-x_pos4[i])*0.1155
    dy4[i] = (y_pos4[i+1]-y_pos4[i])*0.1155
    dr_sq4[i] = dx4[i]**2+dy4[i]**2

dr_sq = np.append(dr_sq1, dr_sq2)
dr_sq = np.append(dr_sq, dr_sq3)
dr_sq = np.append(dr_sq, dr_sq4)
dr_sq = np.sort(dr_sq)
error = np.zeros(len(dr_sq))
for i in range(0, len(error)):
    error[i] = 0.1

dr = np.sqrt(dr_sq)

bins = np.linspace(0, 10, 477) 
bin2 = np.linspace(0, 1, 477)

plt.figure(figsize = (10,5))
p, b = np.histogram(dr, bin2, density=True)
weights = np.ones_like(dr)/float(len(dr))
plt.hist(dr, weights=weights)

#Curve fitting
popt , pcov = curve_fit(rayleigh, dr, p, (1), error, True)

#max likelihood
x = 0
uncertainty = []
for i in dr:
    x += i**2
    uncertainty.append((2*i*0.1)**2)
x = x/(2*len(dr))

max_likeli = boltzmann(x)
ray_est = boltzmann(popt[0])
avo_max = avogadro(max_likeli)
avo_ray = avogadro(ray_est)
print(max_likeli)
print((max_likeli-1.38064852*10**-23)/(1.38064852*10**-23)*100, '%')
print(ray_est)
print((ray_est-1.38064852*10**-23)/(1.38064852*10**-23)*100, '%')
print(avo_max)
print((avo_max-6.0221409*10**23)/(6.0221409*10**23)*100, '%')
print(avo_ray)
print((avo_ray-6.0221409*10**23)/(6.0221409*10**23)*100, '%')

plt.plot(bins, rayleigh(bins, popt), label='Rayleigh Estimation')
plt.plot(bins, rayleigh(bins, x), label='Maximum likelihood estimation')
plt.xlabel("Steps (micrometer)") 
plt.ylabel("Probability")
plt.legend() 

###############################################################################
u_max_likeli = np.sqrt(sum(uncertainty))/(2*len(dr))
u_x = boltzmann_err(x, u_max_likeli)
u_ray = boltzmann_err(popt[0], np.sqrt(pcov[0]))
print()
print('Errors')
print(u_x)
print(u_ray)
print(avo_err(max_likeli, u_x))
print(avo_err(ray_est, u_ray))




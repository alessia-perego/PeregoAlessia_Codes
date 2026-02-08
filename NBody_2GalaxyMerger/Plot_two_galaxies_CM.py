import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from astropy import constants as const
import time


#CORREGGERE CM USARE COSO CHE CALCOLA TIPO MAN MANO (RIGUARDO APP LEZIONE)

#https://www.desmos.com/calculator?lang=it (link for desmos plot tidal condition plot)

###########################################
# UNITS
my_u = ["kpc","yr","M_sun"]

# physical constant converted to chosen units
G = const.G.to(my_u[0]+"^3/("+my_u[1]+"^2 "+my_u[2]+")").value
c = const.c.to(my_u[0]+"/"+my_u[1]).value
print("G =",G,"c =",c)

M0=1 #solar mass
R0=1 #1 kpc
t_int=np.sqrt(R0**3/(G*M0))

###########################################

path = "./" #path to out file
file = path + "out"

N_1=5000 #number of particles used
N_2=500
N=N_1+N_2

n = 3*N + 3 # 'block' length
m = 3    # number of rows to skip (i.e. lines containing n. bodies and time)

time_rows = []
mass_rows = []
data_pos_rows = []
data_vel_rows = []

def read(file, n, m):
	with open(file, "r") as file:
			lines = file.readlines()
			for i in range(0, len(lines), n):  # move through rows every n rows
				time_rows.extend(lines[i+2:i+m])
				mass_rows.extend(lines[i+m:i+m+N])
				data_pos_rows.extend(lines[i+m+N:i+m+2*N])
				data_vel_rows.extend(lines[i+m+2*N:i+m+3*N])

	t = np.loadtxt(time_rows)
	mass = np.loadtxt(mass_rows)
	r = np.loadtxt(data_pos_rows)
	v = np.loadtxt(data_vel_rows)
	
	return t, mass, r, v


####################################
#DATA

t, mass, r, v = read(file, n, m)
x_tot=r[:,0]
y_tot=r[:,1]
z_tot=r[:,2]

Mtot_1=5e9
Mtot_2=5e6
Mtot= np.sum(mass)

#save in each row the evolution of one particle
x_i = np.array([x_tot[j::N] for j in range(N)])
y_i = np.array([y_tot[j::N] for j in range(N)])
z_i = np.array([z_tot[j::N] for j in range(N)])

del x_tot
del y_tot
del z_tot

#norm
d = np.linalg.norm(r, axis=1)
d_i= np.array([d[j::N] for j in range(N)])

print('Data obtained')
####################################################################
#CM

#computing the cm (ONLY FOR GAL1)
r_cm = np.zeros(len(t))
x_cm = np.zeros(len(t))
y_cm = np.zeros(len(t))
z_cm = np.zeros(len(t))
for j in range(len(t)):
	sum_d = 0
	sum_x = 0
	sum_y = 0
	sum_z = 0
	for i in range(N_1):
		sum_d += (mass[i]*d[i+j*N_1])/Mtot_1
		sum_x += (mass[i]*x_i[i][j])/Mtot_1
		sum_y += (mass[i]*y_i[i][j])/Mtot_1
		sum_z += (mass[i]*z_i[i][j])/Mtot_1
	r_cm[j]=(sum_d)
	x_cm[j]=(sum_x)
	y_cm[j]=(sum_y)
	z_cm[j]=(sum_z)

#correcting the coordinates for the cm
r_rel = np.abs(d_i - r_cm[None, :]) 
x_rel = x_i - x_cm[None, :]         
y_rel = y_i - y_cm[None, :]         
z_rel = z_i - z_cm[None, :]  

#del x_i
#del y_i
#del z_i
#del d_i

print('CM done')

'''
fig_r, ax_r = plt.subplots()
ax_r.scatter(t*t_int, r_cm, s=0.6)
ax_r.set_title('CM radius evolution in time')
ax_r.set_xlabel('Time t [yr]')
ax_r.set_ylabel('Radius r [kpc]')
plt.show()
'''

print('time:', len(t))
r_lin=np.linspace(0.0001, 20*np.max(d[0]), 700)

r_cm_r = np.zeros(len(t))
x_cm_r = np.zeros(len(t))
y_cm_r = np.zeros(len(t))
z_cm_r = np.zeros(len(t))


for k in range(len(t)):
	r_cm_lin = np.zeros(len(r_lin))
	x_cm_lin = np.zeros(len(r_lin))
	y_cm_lin = np.zeros(len(r_lin))
	z_cm_lin = np.zeros(len(r_lin))

	for j in range(len(r_lin)):
		sum_d = 0
		sum_x = 0
		sum_y = 0
		sum_z = 0
		for i in range(N_1):
			if (d[i+k*N])<(r_lin[j]):
				sum_d += (mass[i]*d[i+k*N])/Mtot_1
				sum_x += (mass[i]*x_i[i][k])/Mtot_1
				sum_y += (mass[i]*y_i[i][k])/Mtot_1
				sum_z += (mass[i]*z_i[i][k])/Mtot_1
		r_cm_lin[j]=(sum_d)
		x_cm_lin[j]=(sum_x)
		y_cm_lin[j]=(sum_y)
		z_cm_lin[j]=(sum_z)

	for h in range(len(r_lin)):
		if (h+1!=len(r_lin)):
			if (((r_cm_lin[h+1]-r_cm_lin[h])/r_cm_lin[h])<=0.01):
				r_cm_r[k]=r_cm_lin[h]

	print(k, r_cm_r[k])


r_cm_r_tot = np.zeros(len(t))
x_cm_r_tot = np.zeros(len(t))
y_cm_r_tot = np.zeros(len(t))
z_cm_r_tot = np.zeros(len(t))

for k in range(len(t)):
	r_cm_lin = np.zeros(len(r_lin))
	x_cm_lin = np.zeros(len(r_lin))
	y_cm_lin = np.zeros(len(r_lin))
	z_cm_lin = np.zeros(len(r_lin))

	for j in range(len(r_lin)):
		sum_d = 0
		sum_x = 0
		sum_y = 0
		sum_z = 0
		for i in range(N):
			if (d[i+k*N])<(r_lin[j]):
				sum_d += (mass[i]*d[i+k*N])/Mtot
				sum_x += (mass[i]*x_i[i][k])/Mtot
				sum_y += (mass[i]*y_i[i][k])/Mtot
				sum_z += (mass[i]*z_i[i][k])/Mtot
		r_cm_lin[j]=(sum_d)
		x_cm_lin[j]=(sum_x)
		y_cm_lin[j]=(sum_y)
		z_cm_lin[j]=(sum_z)

	for h in range(len(r_lin)):
		if (h+1!=len(r_lin)):
			if (((r_cm_lin[h+1]-r_cm_lin[h])/r_cm_lin[h])<=0.01):
				r_cm_r_tot[k]=r_cm_lin[h]

	print(k, r_cm_r_tot[k])


'''
	fig_r, ax_r = plt.subplots()
	ax_r.scatter(r_lin, r_cm_lin, s=0.6, label='cm_r')
	ax_r.set_xlabel('Time t [yr]')
	ax_r.set_ylabel('Radius r [kpc]')
	plt.show()
'''
print('CM with r computed')

r_rel_CM = np.abs(d_i - r_cm_r[None, :] - r_cm_r_tot[None, :]) 

fig_r, ax_r = plt.subplots()
ax_r.scatter(t, r_cm_r, s=0.6, label='cm_r')
ax_r.scatter(t, r_cm_r_tot, s=0.6, label='cm_r_tot')
ax_r.scatter(t, r_cm, s=0.6, label='cm')
ax_r.set_title('CM radius evolution in time (all part)')
ax_r.set_xlabel('Time t [yr]')
ax_r.set_ylabel('Radius r [kpc]')
plt.legend()
plt.savefig('Radius_CM.png', dpi=300)

##############################################################

#LAGRANGIAN RADII

# Sort stars by distance

#len(r_rel) = particles , len(r_rel[0]) = time

L_r=np.sort(r_rel_CM, axis=0)
fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
lagrangian_radii=[]
for i in range(len(fractions)):
	lagrangian_radii.append(L_r[int(fractions[i]*N-1)])

lagrangian_radii=np.array(lagrangian_radii)

print('Lagrangian radii done')

###############################################################
#PLOT

fig_r, ax_r = plt.subplots()
for i in range(len(fractions)):
    ax_r.scatter(t*t_int, lagrangian_radii[i]*R0, s=0.1)
ax_r.set_title('Radius evolution in time')
ax_r.set_xlabel('Time t [yr]')
ax_r.set_ylabel('Radius r [kpc]')
plt.savefig('Radius_L_CM.png', dpi=300)


################################################
#ANIMATION
# Define figure and 3D axis
fig_anim = plt.figure()
ax_anim = fig_anim.add_subplot(projection='3d')

ax_anim.set_xlim(-100, 100)# (np.min(x_rel), np.max(x_rel))
ax_anim.set_ylim(-100, 100) #(np.min(y_rel),np.max(y_rel))
ax_anim.set_zlim(-100, 100) #(np.min(z_rel), np.max(z_rel))
ax_anim.set_title("Particles evolution in time")
ax_anim.set_xlabel('x [kpc]')
ax_anim.set_ylabel('y [kpc]')
ax_anim.set_zlabel('z [kpc]')

# Scatter plot of initial positions
initial_1 = ax_anim.scatter(x_rel[:N_1,0], y_rel[:N_1,0], z_rel[:N_1,0], s=0.1, color='blue')
initial_2 = ax_anim.scatter(x_rel[N_1:N,0], y_rel[N_1:N,0], z_rel[N_1:N,0], s=5, color='red')

# Function to update animation at each frame
def update(frame):    
    initial_1._offsets3d = (x_rel[:N_1,frame], y_rel[:N_1,frame], z_rel[:N_1,frame])
    initial_2._offsets3d = (x_rel[N_1:N,frame], y_rel[N_1:N,frame], z_rel[N_1:N,frame])
    return initial_1, initial_2

animation = FuncAnimation(fig_anim, update, frames=len(t), interval=30, blit=False) #number of frames is the len of time/ integration step

from matplotlib.animation import FFMpegWriter
animation.save('Plummer_animation.mp4', writer='ffmpeg', fps=30)
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
from astropy.io import fits
import time

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

a_1=10
a_2=1
N_1=5000 #number of particles used
N_2=500
N=N_1+N_2

n = 3*N + 3 # 'block' length
m = 3    # number of rows to skip (i.e. lines containing n. bodies and time)

time_rows = []
mass_rows = []
data_pos_rows = []
data_vel_rows = []

def read(file, n, m, l):
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

t, mass, r, v = read(file, n, m, 1)
x_tot=r[:,0]
y_tot=r[:,1]
z_tot=r[:,2]
vx_tot=v[:,0]
vy_tot=v[:,1]
vz_tot=v[:,2]
Mtot_1=1e6*N_1
Mtot_2=1e5*N_2
Mtot= np.sum(mass)

#save in each row the evolution of one particle
x_i = np.array([x_tot[j::N] for j in range(N)])
y_i = np.array([y_tot[j::N] for j in range(N)])
z_i = np.array([z_tot[j::N] for j in range(N)])

vx_i = np.array([vx_tot[j::N] for j in range(N)])
vy_i = np.array([vy_tot[j::N] for j in range(N)])
vz_i = np.array([vz_tot[j::N] for j in range(N)])

del x_tot
del y_tot
del z_tot
del vx_tot
del vy_tot
del vz_tot

#norm
d = np.linalg.norm(r, axis=1)
d_i= np.array([d[j::N] for j in range(N)])

v_norm = np.linalg.norm(v, axis=1)
v_i= np.array([v_norm[j::N] for j in range(N)])

print('Data obtained')

####################################################################
#CM

#computing the cm for galaxy 1 with iterations 

print('time:', len(t))

r_lin=np.linspace(0.0001, 40*a_1, 400)

r_cm_r = np.zeros(len(t))
x_cm_r = np.zeros(len(t))
y_cm_r = np.zeros(len(t))
z_cm_r = np.zeros(len(t))

r_cm_r_small = np.zeros(len(t))
x_cm_r_small = np.zeros(len(t))
y_cm_r_small = np.zeros(len(t))
z_cm_r_small = np.zeros(len(t))
r_lin_best = np.zeros(len(t))

r_cm_r_tot = np.zeros(len(t))
x_cm_r_tot = np.zeros(len(t))
y_cm_r_tot = np.zeros(len(t))
z_cm_r_tot = np.zeros(len(t))

for k in range(len(t)):
	r_cm_lin = np.zeros(len(r_lin))
	x_cm_lin = np.zeros(len(r_lin))
	y_cm_lin = np.zeros(len(r_lin))
	z_cm_lin = np.zeros(len(r_lin))

	r_cm_lin_2 = np.zeros(len(r_lin))
	x_cm_lin_2 = np.zeros(len(r_lin))
	y_cm_lin_2 = np.zeros(len(r_lin))
	z_cm_lin_2 = np.zeros(len(r_lin))

	r_cm_lin_tot = np.zeros(len(r_lin))
	x_cm_lin_tot = np.zeros(len(r_lin))
	y_cm_lin_tot = np.zeros(len(r_lin))
	z_cm_lin_tot = np.zeros(len(r_lin))

	flag=0
	flag_1=0
	flag_2=0

	for j in range(len(r_lin)):
		sum_d = 0
		sum_x = 0
		sum_y = 0
		sum_z = 0

		sum_d_2 = 0
		sum_x_2 = 0
		sum_y_2 = 0
		sum_z_2 = 0

		for i in range(N_1):
			if (d[i+k*N])<(r_lin[j]):
				sum_d += (mass[i]*d[i+k*N])
				sum_x += (mass[i]*x_i[i][k])
				sum_y += (mass[i]*y_i[i][k])
				sum_z += (mass[i]*z_i[i][k])
		r_cm_lin[j]=(sum_d)/Mtot_1
		x_cm_lin[j]=(sum_x)/Mtot_1
		y_cm_lin[j]=(sum_y)/Mtot_1
		z_cm_lin[j]=(sum_z)/Mtot_1

		for i in range(N_1, N):
			if (d[i+k*N])<(r_lin[j]):
				sum_d_2 += (mass[i]*d[i+k*N])
				sum_x_2 += (mass[i]*x_i[i][k])
				sum_y_2 += (mass[i]*y_i[i][k])
				sum_z_2 += (mass[i]*z_i[i][k])
		r_cm_lin_2[j]=(sum_d_2)/Mtot_2
		x_cm_lin_2[j]=(sum_x_2)/Mtot_2
		y_cm_lin_2[j]=(sum_y_2)/Mtot_2
		z_cm_lin_2[j]=(sum_z_2)/Mtot_2

		r_cm_lin_tot[j]=(sum_d+sum_d_2)/Mtot
		x_cm_lin_tot[j]=(sum_x+sum_x_2)/Mtot
		y_cm_lin_tot[j]=(sum_y+sum_y_2)/Mtot
		z_cm_lin_tot[j]=(sum_z+sum_z_2)/Mtot

		if (j!=0):
			if (((r_cm_lin[j]-r_cm_lin[j-1])/r_cm_lin[j])<=0.01)and(flag==0):
				r_cm_r[k]=r_cm_lin[j]
				x_cm_r[k] = x_cm_lin[j]
				y_cm_r[k] = y_cm_lin[j]
				z_cm_r[k] = z_cm_lin[j]
				flag=1

			if (((r_cm_lin_2[j]-r_cm_lin_2[j-1])/r_cm_lin_2[j])<=0.01)and(flag_1==0):
				r_cm_r_small[k]=r_cm_lin_2[j]
				r_lin_best[k]=r_lin[j]
				x_cm_r_small[k] = x_cm_lin_2[j]
				y_cm_r_small[k] = y_cm_lin_2[j]
				z_cm_r_small[k] = z_cm_lin_2[j]
				flag_1=1

			if (((r_cm_lin_tot[j]-r_cm_lin_tot[j-1])/r_cm_lin_tot[j])<=0.01)and(flag_2==0):
				r_cm_r_tot[k]=r_cm_lin_tot[j]
				x_cm_r_tot[k] = x_cm_lin_tot[j]
				y_cm_r_tot[k] = y_cm_lin_tot[j]
				z_cm_r_tot[k] = z_cm_lin_tot[j]
				flag_2=1

	print(k, r_cm_r[k])

print('CM done')



v_cm = np.zeros(len(t))
vx_cm  = np.zeros(len(t))
vy_cm  = np.zeros(len(t))
vz_cm  = np.zeros(len(t))

v_cm_small = np.zeros(len(t))
vx_cm_small  = np.zeros(len(t))
vy_cm_small  = np.zeros(len(t))
vz_cm_small  = np.zeros(len(t))

for j in range(len(t)):
	sum_d = 0
	sum_x = 0
	sum_y = 0
	sum_z = 0

	for i in range(0, N_1):
		sum_d += (mass[i]*v_norm[i+j*N])
		sum_x += (mass[i]*vx_i[i][j])
		sum_y += (mass[i]*vy_i[i][j])
		sum_z += (mass[i]*vz_i[i][j])
	v_cm [j]=(sum_d)/Mtot_1
	vx_cm[j]=(sum_x)/Mtot_1
	vy_cm [j]=(sum_y)/Mtot_1
	vz_cm [j]=(sum_z)/Mtot_1

	sum_d_2 = 0
	sum_x_2 = 0
	sum_y_2 = 0
	sum_z_2 = 0
	for i in range(N_1, N):
		if(d[i+j*N]<r_lin_best[j]):
			sum_d_2 += (mass[i]*v_norm[i+j*N])
			sum_x_2 += (mass[i]*vx_i[i][j])
			sum_y_2 += (mass[i]*vy_i[i][j])
			sum_z_2 += (mass[i]*vz_i[i][j])
	v_cm_small [j]=(sum_d_2)/Mtot_2
	vx_cm_small [j]=(sum_x_2)/Mtot_2
	vy_cm_small [j]=(sum_y_2)/Mtot_2
	vz_cm_small [j]=(sum_z_2)/Mtot_2


# single primary
primary_hdu = fits.PrimaryHDU()

# Particle time-series as Image HDUs (each receives a 2D numpy array)
hdu_d   = fits.ImageHDU(data=np.asarray(d_i),  name='D_PARTICLES')
hdu_x   = fits.ImageHDU(data=np.asarray(x_i),  name='X_PARTICLES')
hdu_y   = fits.ImageHDU(data=np.asarray(y_i),  name='Y_PARTICLES')
hdu_z   = fits.ImageHDU(data=np.asarray(z_i),  name='Z_PARTICLES')
hdu_v   = fits.ImageHDU(data=np.asarray(v_i),  name='V_PARTICLES')
hdu_vx  = fits.ImageHDU(data=np.asarray(vx_i), name='VX_PARTICLES')
hdu_vy  = fits.ImageHDU(data=np.asarray(vy_i), name='VY_PARTICLES')
hdu_vz  = fits.ImageHDU(data=np.asarray(vz_i), name='VZ_PARTICLES')

# Center-of-mass time series as a single BinTableHDU (1 row per time)
cols_cm = [
    fits.Column(name='t',            array=np.asarray(t),           format='D'),
    fits.Column(name='r_cm_big',     array=np.asarray(r_cm_r),      format='D'),
    fits.Column(name='x_cm_big',     array=np.asarray(x_cm_r),      format='D'),
    fits.Column(name='y_cm_big',     array=np.asarray(y_cm_r),      format='D'),
    fits.Column(name='z_cm_big',     array=np.asarray(z_cm_r),      format='D'),
	fits.Column(name='v_cm_big',     array=np.asarray(v_cm),		format='D'),
    fits.Column(name='vx_cm_big',    array=np.asarray(vx_cm),       format='D'),
    fits.Column(name='vy_cm_big',    array=np.asarray(vy_cm),       format='D'),
    fits.Column(name='vz_cm_big',    array=np.asarray(vz_cm),       format='D'),
    fits.Column(name='r_cm_small',   array=np.asarray(r_cm_r_small),format='D'),
    fits.Column(name='x_cm_small',   array=np.asarray(x_cm_r_small),format='D'),
    fits.Column(name='y_cm_small',   array=np.asarray(y_cm_r_small),format='D'),
    fits.Column(name='z_cm_small',   array=np.asarray(z_cm_r_small),format='D'),
    fits.Column(name='r_lin_best',   array=np.asarray(r_lin_best),  format='D'),
	fits.Column(name='v_cm_small',   array=np.asarray(v_cm_small),  format='D'),
    fits.Column(name='vx_cm_small',  array=np.asarray(vx_cm_small), format='D'),
    fits.Column(name='vy_cm_small',  array=np.asarray(vy_cm_small), format='D'),
    fits.Column(name='vz_cm_small',  array=np.asarray(vz_cm_small), format='D'),
    fits.Column(name='r_cm_tot',     array=np.asarray(r_cm_r_tot),   format='D'),
    fits.Column(name='x_cm_tot',     array=np.asarray(x_cm_r_tot),   format='D'),
    fits.Column(name='y_cm_tot',     array=np.asarray(y_cm_r_tot),   format='D'),
    fits.Column(name='z_cm_tot',     array=np.asarray(z_cm_r_tot),   format='D'),
]
hdu_cm = fits.BinTableHDU.from_columns(cols_cm, name='CENTER_OF_MASS')

# write file
hdul = fits.HDUList([primary_hdu,
                     hdu_d, hdu_x, hdu_y, hdu_z,
                     hdu_v, hdu_vx, hdu_vy, hdu_vz,
                     hdu_cm])
hdul.writeto('simulation_results.fits', overwrite=True)
print('fits done')


r_rel = np.abs(d_i - r_cm_r[None, :] - r_cm_r_tot[None, :]) 
x_rel = x_i - x_cm_r[None, :] - x_cm_r_tot[None, :]   
y_rel = y_i - y_cm_r[None, :] - y_cm_r_tot[None, :]       
z_rel = z_i - z_cm_r[None, :] - z_cm_r_tot[None, :] 

#correcting the coordinates for the cm
r_rel_small = np.abs(d_i[N_1:N] - r_cm_r_small[None, :] - r_cm_r[None, :] - r_cm_r_tot[None,:]) 
x_rel_small = x_i[N_1:N] - x_cm_r_small[None, :] - x_cm_r[None, :] - x_cm_r_tot[None, :]       
y_rel_small = y_i[N_1:N] - y_cm_r_small[None, :] - y_cm_r[None, :] - y_cm_r_tot[None, :]              
z_rel_small = z_i[N_1:N] - z_cm_r_small[None, :] - z_cm_r[None, :] - z_cm_r_tot[None, :] 

print('CM tot done')

r_cm_small_components=np.stack([x_cm_r_small, y_cm_r_small, z_cm_r_small], axis=1)
v_cm_small_components=np.stack([vx_cm_small, vy_cm_small, vz_cm_small], axis=1)
print(r_cm_small_components.shape, v_cm_small_components.shape )
l = np.cross(r_cm_small_components, v_cm_small_components)
L = np.linalg.norm(l, axis=1) * Mtot_2

print('v CM done')


del x_i
del y_i
del z_i
del d_i

L0 = L[0]
threshold = 0.1 * L0
idx_decay_max = np.argmax(L < threshold)
idx_decay = np.where(L < threshold)[0]

for i in range(len(idx_decay)):
	if L[idx_decay[i]] < threshold:
		t_decay = t[idx_decay[i]]*t_int / 1e9  # Tempo in Gyr
		print(f"Dynamical Friction Timescale (L < 10% L0): {t_decay:.2f} Gyr")
	else:
		print("L > 10% L0 always")

if L[idx_decay_max] < threshold:
	t_decay = t[idx_decay_max]*t_int / 1e9  # Tempo in Gyr
	print(f"Dynamical Friction Timescale (L < 10% L0): {t_decay:.2f} Gyr")


fig_r, ax_r = plt.subplots()
ax_r.scatter(t, r_cm_r, s=0.6, label='cm_r')
ax_r.scatter(t, r_cm_r_tot, s=0.6, label='cm_r_tot')
ax_r.scatter(t, r_cm_r_small, s=0.6, label='cm_r_small')
ax_r.set_title('CM radius evolution in time (all part)')
ax_r.set_xlabel('Time t [yr]')
ax_r.set_ylabel('Radius r [kpc]')
plt.legend()
plt.savefig('Radius_CM.png', dpi=300)

fig_r, ax_r = plt.subplots()
ax_r.scatter(t*t_int, r_cm_r_small, s=0.1)
ax_r.set_title('Radious CM small evolution in time')
ax_r.set_xlabel('Time t [yr]')
ax_r.set_ylabel('Radious r [kpc]')
plt.savefig('Radious_CM_small.png', dpi=300)

fig_l, ax_l = plt.subplots()
ax_l.scatter(t*t_int, L/(R0**2/t_int), s=0.1)
ax_l.set_title('Angular momentum CM small evolution in time')
ax_l.set_xlabel('Time t [yr]')
ax_l.set_ylabel('L [Kpc^2/yr]')
plt.savefig('Momentum_CM_small.png', dpi=300)

##############################################################

#LAGRANGIAN RADII

# Sort stars by distance

#len(r_rel) = particles , len(r_rel[0]) = time

L_r=np.sort(r_rel, axis=0)
fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
lagrangian_radii=[]
for i in range(len(fractions)):
	lagrangian_radii.append(L_r[int(fractions[i]*N-1)])

lagrangian_radii=np.array(lagrangian_radii)

print('Lagrangian radii done')

###############################################################
#PLOT
'''
#plot for positions in 3d
fig_3D = plt.figure()
ax_3D = fig_3D.add_subplot(projection='3d')
for i in range(N):
	ax_3D.plot(x_rel[i]*R0,y_rel[i]*R0,z_rel[i]*R0)
ax_3D.set_title('Positions of all the particles')
ax_3D.set_xlabel('x [kpc]')
ax_3D.set_ylabel('y [kpc]')
ax_3D.set_zlabel('z [kpc]')
plt.savefig('Positions_3D.png', dpi=300)
'''


fig_r, ax_r = plt.subplots()
for i in range(len(fractions)):
    ax_r.scatter(t*t_int, lagrangian_radii[i]*R0, s=0.1)
ax_r.set_title('Radius evolution in time')
ax_r.set_xlabel('Time t [yr]')
ax_r.set_ylabel('Radius r [kpc]')
plt.savefig('Radius_L.png', dpi=300)

'''
#plot for radius in time
fig_r, ax_r = plt.subplots()
for i in range(N_1,N):
    ax_r.scatter(t*t_int, r_rel[i]*R0, s=0.1)
ax_r.set_title('Radius evolution in time')
ax_r.set_xlabel('Time t [yr]')
ax_r.set_ylabel('Radius r [kpc]')
plt.savefig('Radius_small.png', dpi=300)
'''

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
initial_1 = ax_anim.scatter(x_rel[:N_1,0], y_rel[:N_1,0], z_rel[:N_1,0], s=0.05, color='blue')
initial_2 = ax_anim.scatter(x_rel[N_1:N,0], y_rel[N_1:N,0], z_rel[N_1:N,0], s=4, color='red')

# Function to update animation at each frame
def update(frame):    
    initial_1._offsets3d = (x_rel[:N_1,frame], y_rel[:N_1,frame], z_rel[:N_1,frame])
    initial_2._offsets3d = (x_rel[N_1:N,frame], y_rel[N_1:N,frame], z_rel[N_1:N,frame])
    return initial_1, initial_2

animation = FuncAnimation(fig_anim, update, frames=len(t), interval=30, blit=False) #number of frames is the len of time/ integration step
#from IPython.display import HTML
#from IPython.display import display
#display(HTML(animation.to_html5_video()))

from matplotlib.animation import FFMpegWriter

#writer = FFMpegWriter(fps=15, bitrate=1800)
#animation.save("dynamical_friction_8.mp4", writer=writer)
animation.save('Plummer_animation.mp4', writer='ffmpeg', fps=30)
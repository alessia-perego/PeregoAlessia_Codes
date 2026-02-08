import numpy as np
from astropy import constants as const
from astropy import units as u
import matplotlib
from matplotlib import _docstring
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.integrate import quad

#SYSTEM 1: ./treecode in=in out=out dtime=0.03 eps=0.04 theta=0.1 tstop=10 dtout=0.1

# file name, path and options
path_out = "./../Treecode/" # example path
file_name = "in" # example name
################################################################################

# UNITS
my_u = ["kpc","yr","M_sun"]

# physical constant converted to chosen units
G = const.G.to(my_u[0]+"^3/("+my_u[1]+"^2 "+my_u[2]+")").value
c = const.c.to(my_u[0]+"/"+my_u[1]).value
print("G =",G,"c =",c)


# conversion constants between physical and internal units
R0 = 1.
M0 = 1.
T0 = np.sqrt(R0**3/(G*M0))
################################################################################
# PARTICLE INITIALIZATION
dim=3 

r_perc=0.9

#GALAXY 1 (BIG)
N_1=10000

m_particles_1 = 1e6*np.ones(N_1)
a_1 = 10
M_tot_1 = np.sum(m_particles_1)


#GALAXY 2 (SMALL)
N_2=1000

m_particles_2 = 1e5*np.ones(N_2)
a_2 = 1
M_tot_2 = np.sum(m_particles_2)

#dynamical_t=(2*np.pi*np.sqrt(a**3/(G*M_tot)))/T0
#print('Dynamical time in internal units: ', dynamical_t)

#space and velocity angles

#GALAXY 1
t_s_1=np.arccos(1-2*np.random.uniform(0,1,N_1))
t_v_1=np.arccos(1-2*np.random.uniform(0,1,N_1))
fi_s_1=np.random.uniform(0,1,N_1)*2*np.pi
fi_v_1=np.random.uniform(0,1,N_1)*2*np.pi

r_s_1=a_1/np.sqrt((np.random.uniform(0,1,N_1)**(-2/3)-1)) #Radius is sampled from the inverse of the mass distribution

r1_1=np.zeros((N_1,3))

#spatial components
for i in range(0,N_1):
	x_1=r_s_1[i]*np.sin(t_s_1[i])*np.cos(fi_s_1[i])
	y_1=r_s_1[i]*np.sin(t_s_1[i])*np.sin(fi_s_1[i])
	z_1=r_s_1[i]*np.cos(t_s_1[i])
	r1_1[i]=np.array([x_1,y_1,z_1])


#90% MASS RADIOUS
def m_plummer_1(r):
  return M_tot_1*(r**3/(r**2+a_1**2)**(3/2))

def m_in(p):
  m_r=0
  r_cycle=np.linspace(0,150*a_1,10000)
  for i in r_cycle:
    m_r=m_plummer_1(i)
    if (m_r>p*M_tot_1):
      return i

#perturber at r so that inside about 90% of mass
r_shift=m_in(r_perc)
print('Shift radious: ', r_shift)


#GALAXY 2
t_s_2=np.arccos(1-2*np.random.uniform(0,1,N_2))
t_v_2=np.arccos(1-2*np.random.uniform(0,1,N_2))
fi_s_2=np.random.uniform(0,1,N_2)*2*np.pi
fi_v_2=np.random.uniform(0,1,N_2)*2*np.pi

r_s_2=a_2/np.sqrt((np.random.uniform(0,1,N_2)**(-2/3)-1)) #Radius is sampled from the inverse of the mass distribution

r1_2=np.zeros((N_2,3))

#spatial components
for i in range(0,N_2):
	x_2=(r_s_2[i])*np.sin(t_s_2[i])*np.cos(fi_s_2[i])+r_shift
	y_2=(r_s_2[i])*np.sin(t_s_2[i])*np.sin(fi_s_2[i])
	z_2=(r_s_2[i])*np.cos(t_s_2[i])
	r1_2[i]=np.array([x_2,y_2,z_2])

#GALAXY 1
def potential_1(r): #psi = -fi+fi0
	return -(G)*M_tot_1/(np.sqrt((a_1)**2+r**2)) #Plummer potential

def P(q):
	return (1-q**2)**(7/2)*q**2

#Normalize the distribution over [0, 1] ---
normalization, _ = quad(P, 0, 1)

def norm_P(q):
    return P(q) / normalization

q_grid = np.linspace(0, 1, 10000)
max_f = np.max(norm_P(q_grid))

#0.01 max for q distribution
#try and catch

x_tc_1=[]
y_tc_1=[]
v_1=[]

for i in range(100000000):
	tac_x_1=random.uniform(0,1)
	tac_y_1=random.uniform(0,max_f)
	if tac_y_1<P(tac_x_1):
		x_tc_1.append(tac_x_1)
		y_tc_1.append(tac_y_1)
	if len(x_tc_1)==(N_1):
		break

#given a value of x following q distirbution: v = x*vescape = x*sqrt(-2*potential)
for i in range(N_1):
	v_1.append((x_tc_1[i]*np.sqrt(-2*potential_1(r_s_1[i])))) #given a velocity = i associate it to a particle in a given position => v_escape of that position
v_1=np.array(v_1)


#bulk v for 2 galaxy
v_circ=np.sqrt((m_plummer_1(m_in(r_perc)))/m_in(r_perc))
print('Circular velocity: ',v_circ)


#velocity components
v1_1=np.zeros((N_1,3))
for i in range(0,N_1):
	vx_1=(v_1[i]*np.sin(t_v_1[i])*np.cos(fi_v_1[i]))/ (R0/T0)
	vy_1=(v_1[i]*np.sin(t_v_1[i])*np.sin(fi_v_1[i]))/ (R0/T0)-v_circ*10**(-2)
	vz_1=(v_1[i]*np.cos(t_v_1[i]))/ (R0/T0)
	v1_1[i] = np.array([vx_1,vy_1,vz_1])


#GALAXY 2
def potential_2(r): #psi = -fi+fi0
	return -(G)*M_tot_2/(np.sqrt((a_2)**2+r**2)) #Plummer potential

#0.01 max for q distribution
#try and catch
x_tc_2=[]
y_tc_2=[]
v_2=[]

for i in range(100000000):
	tac_x_2=random.uniform(0,1)
	tac_y_2=random.uniform(0,max_f)
	if tac_y_2<P(tac_x_2):
		x_tc_2.append(tac_x_2)
		y_tc_2.append(tac_y_2)
	if len(x_tc_2)==N_2:
		break

#given a value of x following q distirbution: v = x*vescape = x*sqrt(-2*potential)

for i in range(N_2):
	v_2.append((x_tc_2[i]*np.sqrt(-2*potential_2(r_s_2[i])))) #given a velocity = i associate it to a particle in a given position => v_escape of that position
v_2=np.array(v_2)

#velocity components
v1_2=np.zeros((N_2,3))
for i in range(0,N_2):
	vx_2=(v_2[i]*np.sin(t_v_2[i])*np.cos(fi_v_2[i]))/(R0/T0)
	vy_2=(v_2[i]*np.sin(t_v_2[i])*np.sin(fi_v_2[i]))/(R0/T0)+(v_circ) #all particles put in ciruclar motion in y direction (in x moved system)
	vz_2=(v_2[i]*np.cos(t_v_2[i]))/(R0/T0)
	v1_2[i] = np.array([vx_2,vy_2,vz_2])


################################################################################
# PRINT IC

N=N_1+N_2
m_1 = np.array([m_particles_1])[0] / M0
m_2 = np.array([m_particles_2])[0] / M0
R_1= np.array([r1_1])[0] / R0
R_2= np.array([r1_2])[0] / R0
V_1 = np.array([v1_1])[0] 
V_2 = np.array([v1_2])[0]

######################################

fig0 = plt.figure(figsize=(8,8))
ax0 = fig0.add_subplot(projection='3d')
#ax0.set_title('Example of initial positions in a sphere')
for i in range(0,N_1):
	ax0.scatter(R_1[i][0],R_1[i][1],R_1[i][2], s=0.1, color='black')

for i in range(0,N_2):
	ax0.scatter(R_2[i][0],R_2[i][1],R_2[i][2], s=0.5, color='red')
ax0.set_xlabel('x [kpc]')
ax0.set_ylabel('y [kpc]')
ax0.set_ylabel('z [kpc]')
ax0.set_xlim(-100,100)
ax0.set_ylim(-100,100)
ax0.set_zlim(-100,100)
fig0.suptitle('Initial position of the system in 3D')
plt.tight_layout()
plt.savefig('In_plot.png', dpi=300)
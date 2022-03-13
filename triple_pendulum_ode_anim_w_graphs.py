import numpy as np
import sympy as sym
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation


def a1a2(to12, ti, p):
	theta1,omega1,theta2,omega2,theta3,omega3=to12
	m1,m2,m3,l1,l2,l3,g=p
	
	return [omega1,alpha1.subs({M1:m1,M2:m2,M3:m3,L1:l1,L2:l2,L3:l3,G:g,th1:theta1,th2:theta2,th3:theta3,th1d:omega1,th2d:omega2,th3d:omega3}),\
		omega2,alpha2.subs({M1:m1,M2:m2,M3:m3,L1:l1,L2:l2,L3:l3,G:g,th1:theta1,th2:theta2,th3:theta3,th1d:omega1,th2d:omega2,th3d:omega3}),\
		omega3,alpha3.subs({M1:m1,M2:m2,M3:m3,L1:l1,L2:l2,L3:l3,G:g,th1:theta1,th2:theta2,th3:theta3,th1d:omega1,th2d:omega2,th3d:omega3})]

L1,L2,L3 = sym.symbols('L1 L2 L3')
M1,M2,M3 = sym.symbols('M1 M2 M3')
G = sym.symbols('g')
t = sym.Symbol('t')
th1,th2,th3 = dynamicsymbols('th1 th2 th3')

x1=L1*sym.sin(th1)
x2=L1*sym.sin(th1)+L2*sym.sin(th2)
x3=L1*sym.sin(th1)+L2*sym.sin(th2)+L3*sym.sin(th3)
y1=-L1*sym.cos(th1)
y2=-L1*sym.cos(th1)-L2*sym.cos(th2)
y3=-L1*sym.cos(th1)-L2*sym.cos(th2)-L3*sym.cos(th3)

x1dot=x1.diff(t,1)
x2dot=x2.diff(t,1)
x3dot=x3.diff(t,1)
y1dot=y1.diff(t,1)
y2dot=y2.diff(t,1)
y3dot=y3.diff(t,1)

sum1=x1dot**2+y1dot**2
sum2=x2dot**2+y2dot**2
sum3=x3dot**2+y3dot**2

T=0.5*M1*sum1+0.5*M2*sum2+0.5*M3*sum3
V=M1*G*y1+M2*G*y2+M3*G*y3
L=T-V

ddth1=L.diff(th1,1)
ddth2=L.diff(th2,1)
ddth3=L.diff(th3,1)

ddthd1=L.diff(th1.diff(t,1),1)
ddthd2=L.diff(th2.diff(t,1),1)
ddthd3=L.diff(th3.diff(t,1),1)

ddtddthd1=ddthd1.diff(t,1)
ddtddthd2=ddthd2.diff(t,1)
ddtddthd3=ddthd3.diff(t,1)

lg1=sym.simplify(ddtddthd1-ddth1)
lg2=sym.simplify(ddtddthd2-ddth2)
lg3=sym.simplify(ddtddthd3-ddth3)

th1d=th1.diff(t,1)
th2d=th2.diff(t,1)
th3d=th3.diff(t,1)
th1dd=th1.diff(t,2)
th2dd=th2.diff(t,2)
th3dd=th3.diff(t,2)

sol=sym.solve([lg1,lg2,lg3],(th1dd,th2dd,th3dd))

alpha1=sol[th1dd]
alpha2=sol[th2dd]
alpha3=sol[th3dd]

m1=1
m2=1
m3=1
l1=0.5
l2=0.75
l3=1
thetao1=90
thetao2=135
thetao3=180
omegao1=0
omegao2=0
omegao3=0
g=9.8

cnvrt=np.pi/180

thetao1*=cnvrt
thetao2*=cnvrt
thetao3*=cnvrt

p=[m1,m2,m3,l1,l2,l3,g]
to12=[thetao1,omegao1,thetao2,omegao2,thetao3,omegao3]

tf = 120 
nfps = 30 
nframes = tf * nfps
tt = np.linspace(0, tf, nframes)

aw = odeint(a1a2, to12, tt, args = (p,))

tha1=aw[:,0]
tha2=aw[:,2]
tha3=aw[:,4]

xa1=l1*np.sin(tha1)
ya1=-l1*np.cos(tha1)
xa2=xa1+l2*np.sin(tha2)
ya2=ya1-l2*np.cos(tha2)
xa3=xa2+l3*np.sin(tha3)
ya3=ya2-l3*np.cos(tha3)

lmax=l1+l2+l3+0.2
lmin=-l1-l2-l3-0.2

pe1=m1*g*ya1
pe2=m2*g*ya2
pe3=m3*g*ya3

w1=aw[:,1]
w2=aw[:,3]
w3=aw[:,5]

ke1=0.5*m1*((l1*w1)**2)
ke2=0.5*m2*(((l1*w1)**2)+((l2*w2)**2)+(2*l1*l2*w1*w2*np.cos(tha1-tha2)))
ke3=0.5*m3*(((l1*w1)**2)+((l2*w2)**2)+((l3*w3)**2)+(2*l1*l2*w1*w2*np.cos(tha1-tha2))+(2*l1*l3*w1*w3*np.cos(tha1-tha3))+(2*l2*l3*w2*w3*np.cos(tha2-tha3)))

E1=ke1+pe1
E2=ke2+pe2
E3=ke3+pe3
E=E1+E2+E3

ke=ke1+ke2+ke3
pe=pe1+pe2+pe3

Emax=max(E)

pe1/=Emax
pe2/=Emax
pe3/=Emax
ke1/=Emax
ke2/=Emax
ke3/=Emax
E1/=Emax
E2/=Emax
E3/=Emax
E/=Emax
ke/=Emax
pe/=Emax

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	plt.arrow(0,0,xa1[frame],ya1[frame],head_width=None,color='b')
	circle=plt.Circle((xa1[frame],ya1[frame]),radius=0.05,fc='r')
	plt.gca().add_patch(circle)
	plt.arrow(xa1[frame],ya1[frame],xa2[frame]-xa1[frame],ya2[frame]-ya1[frame],head_width=None,color='b')
	circle=plt.Circle((xa2[frame],ya2[frame]),radius=0.05,fc='r')
	plt.gca().add_patch(circle)
	plt.arrow(xa2[frame],ya2[frame],xa3[frame]-xa2[frame],ya3[frame]-ya2[frame],head_width=None,color='b')
	circle=plt.Circle((xa3[frame],ya3[frame]),radius=0.05,fc='r')
	plt.gca().add_patch(circle)
	plt.title("triple pendulum")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([lmin,lmax])
	plt.ylim([lmin,lmax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(tt[0:frame],ke[0:frame],'r',lw=0.5)
	plt.plot(tt[0:frame],pe[0:frame],'b',lw=0.5)
	plt.plot(tt[0:frame],E[0:frame],'g',lw=1.0)
	plt.xlim([0,tf])
	plt.title("Energy (Rescaled)")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')


ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('trippendode.mp4', writer=writervideo)

plt.show()





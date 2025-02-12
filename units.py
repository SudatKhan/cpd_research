import numpy as np

cspeed = 2.998e10 # speed of light
rada = 7.5657e-15 # radiation constant
kb = 1.380649e-16 # boltzmann constant
G = 6.67430e-8 # gravitational constant
mp = 1.67262192e-24 # proton mass
AU = 1.496e13 # 1 AU in cm
Msun = 1.989e33 # solar mass in g
Mearth = 5.972e27 # earth mass in g
Mjup = 1.899e30 # jupiter mass in g
R = 8.314462618e7 # ideal gas const
Rjup = 6.9911e9 # jupiter radius in cm
Lsun = 3.828e33 # solar luminosity in erg/s

T0 = 66.0 # Kelvin
mu = 2.3 # mean molecular weight
a = 5.0*AU # cm
Mp = 1.0*Mjup # g
Mstar = 1.0*Msun # g
Sigma = 488.0 # g/cm^2
Lp = 1.0e-5*Lsun # solar luminosity
alpha = 0.001 # protoplanetary disk alpha
core_density = 3.0 # g/cm^3

print()
print('Input parameters')
print('----------------------')
print('T0 =', T0, "Kelvin")
print('mu =', mu)
print('a =', a/AU, "AU")
print('Mp =', Mp/Mearth, "earth mass =", Mp/Mjup, "jupiter mass")
print('Mstar =', Mstar/Msun, "solar mass")
print('Sigma =', Sigma, "g/cm^2")
print('alpha =', alpha)
print()

cs = np.sqrt(kb*T0/mu/mp)
Omega0 = np.sqrt(G*Mstar/a**3)
H0 = cs/Omega0
t0 = 1.0/Omega0
K = (Mp/Mstar)**2*(H0/a)**-5/alpha
gap_depth = 1.0/(1.0+0.04*K)
Kprime = (Mp/Mstar)**2*(H0/a)**-3/alpha
gap_width = 0.41*a*Kprime**0.25
Rhill = a*(Mp/Mstar/3.0)**(1.0/3.0)
rho0 = gap_depth*Sigma/np.sqrt(2.0*np.pi)/H0
qthermal = Mp/Mstar/(H0/a)**3
crat = cspeed/cs
prat = rada*(mu*mp/kb)**4/rho0*H0**6/t0**6
rcore = (3.0*Mp/4.0/np.pi/core_density)**(1.0/3.0)
print('Simulation parameters')
print('-----------------------')
print("prat =", prat)
print("crat =", crat)
print()
print("Alternatively use:")
print("tunit =", T0)
print("rhounit =", rho0)
print("lunit =", H0)
print("timeunit =", t0)
print()
print("To give:")
print("prat =", rada*T0*T0*T0/rho0/R*mu)
print("crat =", cspeed/np.sqrt(R/mu*T0))
print()
print("Other parameters")
print("-----------------------")
print("qthermal =", qthermal)
print("H/R =", H0/a)
print("Rjup/H0 =", Rjup/H0)
print("Rhill/H0 =", (qthermal/3.0)**(1.0/3.0))
print("cprat =", crat*prat)
print("gap depth =", gap_depth) # from Kanagawa+2015
print("gapwidth/Rhill = ", gap_width/Rhill) # from Kanagawa+2016
print("core radius/H0 = ", rcore/H0)
print()
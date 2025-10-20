import numpy as np; from scipy.integrate import odeint; import random
import matplotlib.pyplot as plt

def mass_spring_system0D(start, t, m, k):
    #Defines the system of differential equations for a mass-spring system.
    pos,vel=start
    dveldt = -k / m * pos
    return [vel, dveldt]

def mass_spring_system2D(xPos,xVel,yPos,yVel, t, m, k):
        time = np.linspace(0, t, t*100)
        print("xPos,xVel,yPos,yVel, t, m, k)", xPos,xVel,yPos,yVel, t, m, k)
        x = odeint(mass_spring_system0D,[xPos,xVel], time, args=(m, k))
        y = odeint(mass_spring_system0D,[yPos,yVel], time, args=(m, k))
        return x[:,0],x[:,1], y[:,0],y[:,1]

def randomCentered(x):
        return x*random.random()-x, x*random.random()-x

def initialCond():
        t,m,k=random.randint(1,1000),10*random.random(),10*random.random()+1
        xPos,yPos=randomCentered(1000)
        xVel,yVel=randomCentered(100)
        return xPos,xVel,yPos,yVel, t, m, k

def fullSimulation():
        xPos,xVel,yPos,yVel, t, m, k =  initialCond()
        dposXdt, dvelXdt,  dposYdt, dvelYdt=mass_spring_system2D(xPos,xVel,yPos,yVel, t, m, k)
        ends=[dposXdt[-1], dvelXdt[-1],  dposYdt[-1], dvelYdt[-1]]
        starts=[xPos,xVel,yPos,yVel, t, m, k]
        out=(np.append(starts,ends)).flatten()
        return out

out=fullSimulation()

def energy(a,b):
  return 0.5*a*b**2
def springPotentialAndKE(posX,velX,posY,velY,k,m):
  return energy(m,velX)+energy(k,posX)+energy(m,velY)+energy(k,posY)
def physicLoss(out):
  xPos,xVel,yPos,yVel, t, m, k,xPosEnd,xVelEnd,yPosEnd,yVelEnd=out
  begEnergy=springPotentialAndKE(xPos,xVel,yPos,yVel,k,m)
  endEnergy=springPotentialAndKE(xPosEnd,xVelEnd,yPosEnd,yVelEnd,k,m)
  return abs(begEnergy-endEnergy)

#pos = positon, vel = velocity, mag=magnetic 
import numpy as np
import math

def KineticComponet(xVel, yVel):
  return (xVely**2+yVel**2)/2

def magneticPotential(magnField,yPos,xPos,xVel,yVel,charge):
  crossProduct=(yPos*xVel-xPos*yVel)
  magFieldPotential=abs(charge*magField*crossProduct)
  return magFieldPotential
  
def (magField, charge,xPosOrig, yPosnOrig,  xVelOrig, yVelOrig, xPosNow, yPosNow,  xVelNow, yVelNow):
  KCOrig, KCNow=KineticComponet(xVelyOrig, yVelOrig), kineticComponet(xVelNow, yVelNow) 
  magPotentialOrig=magneticPotential(magField,yPosOrig,xPosOrig,xVelOrig,yVelOrig)
  magPotentialNow=magneticPotential(magField,yPosNow,xPosNow,xVelNow,yVelNow)
  diff=KCNow-KCOrig+magPotentialNow-magPotentialOrig
  return abs(diff)

#energy in system must remain constant, that is, potential + kinetic
#This sees if that remains true, mass is ignored because it can be factored out
#magnetic potential comes from charge * MagneticField x Velocity (the Lorentz Force), where cross product is x
#To find the cross product, or to find the component of velocity traveling perpendicular to the magnetic field, 
#if we assume the field is centered at the origin (0,0)
#It is Velocity x Location, dividing by distance from origin would give velocity component in the right direction, but we do not do that as
#that extra distance component converts force to energy
#we need the absolute value of this, as the potential field could be pointing in two different directions towards or away, 
#but we only care about the magnitude
# This loss function makes sure part of physics is followed, but cannot give exact answers



#sine fitting weights


def epochWeightSineForOneWeight(epoch,numberOfLossFunctions,number): 
  #fluctuates weights based on sine curve, making sure always positive, and weights add up to one
  return (math.sin(epoch+2*math.pi*number/numberOfLossFunctions)+1)/2

def epochWeightSineBased(epoch,numberOfLossFunctions): #does weights for all loss functions 
  weightsArray=[]
  for number in range (0, numberOfLossFunctions):
    weightsArray=weightsArray.append(epochWeightSineForOneWeigh(epoch,numberOfLossFunctions,number)
return weightsArray






#curve fitting fancy

def unitVector(x):
        square=np.square(x)
        scale=0
        for z in range(0,len(x)):
                scale=scale+square[z]
        return x/scale

def adaptiveLossNoSin(lossList,weightList):
        diff=lossList[:-1]-lossList[:-2]
        square=np.square(weightList[-1])
        unit=unitVector(square)
        if diff < 0:
                return -1*unit
        return unit

#if weights are improving from before, 
#generally we should increase bigger weights more and smaller weights less as it seems bigger weights more important
#square is to make sure its not linear and for other reasons later explained
#this makes it like you are speeding up or slowing down sine function fluctation used before
#square also makes speed up of individual factors more different, to explore different "offsets" or "phases" better
#does reverse direction if change in loss is zero or negative  
#normalization keeps it bounded
#unitVector function was found to be faster than using pure numpy implentation


def curveFancy(loss, weightsList, epochs,numberOfLossFunctions):
  minTodoFancy=numberOfLossFunctions+2
  if minTodoFancy < 7: minTodoFancy = 7 #makes sure at least one cycle as, as 2Pi rounded up is seven
  if epochs > minTodoFancy: #make sure enough points for curve fitting, ele use sine funciton above to colect more varied points
    adaptiveLossNoSin(lossList,weightList)+epochWeightSineForOneWeight(epoch,numberOfLossFunctions)
    newWeights[newWeights == 0] = 0.1 #make sure no weight is zero, as then no data there and it vanishes
    min=np.min(newWeights)
    if min < 0: newWeights=newWeights+min #no negative weights, as this would make things worse in some direction
    return newWeights=unitVector(newWeights) #normalized, so its bounded and nothing grows to extreme
  else:
    return newWeights=epochWeightSineForOneWeight(epoch,numberOfLossFunctions)

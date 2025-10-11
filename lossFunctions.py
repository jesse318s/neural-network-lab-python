#Weights here refer to weights to balance relative loss functions, like MAE and MSE, not weights used between connections between neurons
#pos = position, vel = velocity, mag=magnetic, vec=vector, func=function 

import numpy as np
import math

def notListOrArray(x): #fix #this function not working right to test if array or list, or int or float of some kind for later functions
	#so ugly try and excpet used instead
  print(type(x),x)
  if (type(x) != list) or (type(x) != np.array) or (type(x) != numpy.ndarray) :
    return False
  try:
    q=x[0]
    return False
  except:
    False
  return True


def unitVec(x):
        squared=np.square(x)
        scale=0
        if notListOrArray(squared):
                for z in range(0,len(x)):
                        scale=scale+squared[z]
        else:
                return 1
        return np.divide(x,scale)

def oneMaker(x): #makes all weights add up to one or negative one assuming all weights have the same sign
        return x/np.sum(np.abs(x))

def KineticComponet(xVel, yVel):
  return (xVel**2+yVel**2)/2

def magneticPotential(magField,yPos,xPos,xVel,yVel,charge):
  crossProduct=(yPos*xVel-xPos*yVel)
  magFieldPotential=abs(charge*magField*crossProduct)
  return magFieldPotential
  
def physicLoss(magField, charge,xPosOrig, yPosOrig,  xVelOrig, yVelOrig, xPosNow, yPosNow,  xVelNow, yVelNow):
  KCOrig, KCNow=KineticComponet(xVelOrig, yVelOrig), KineticComponet(xVelNow, yVelNow) 
  magPotentialOrig=magneticPotential(magField,yPosOrig,xPosOrig,xVelOrig,yVelOrig,charge)
  magPotentialNow=magneticPotential(magField,yPosNow,xPosNow,xVelNow,yVelNow,charge)
  diff=KCNow-KCOrig+magPotentialNow-magPotentialOrig
  return abs(diff)

#energy in system must remain constant, that is, potential + kinetic
#This sees if that remains true; mass is ignored because it can be factored out
#magnetic potential comes from charge * MagneticField x Velocity (the Lorentz Force), where cross product is x
#To find the cross product, or to find the component of velocity traveling perpendicular to the magnetic field, 
#if we assume the field is centered at the origin (0,0)
#It is Velocity x Location, dividing by distance from origin would give velocity component in the right direction, but we do not do that, as
#that extra distance component converts force to energy
#we need the absolute value of this, as the potential field could be pointing in two different directions, towards or away, 
#But we only care about the magnitude of 
# This loss function 


#sine fitting weights


def epochWeightSineForOneWeight(epoch,numberOfLossFuncs,number): 
    #fluctuates weights based on a sine curve, making sure they are always positive, and the weights add up to one
  print("epoch,numberOfLossFuncs,number")
  return (math.sin(epoch+2*math.pi*number/numberOfLossFuncs)+1)/2

def epochWeightSineBased(epoch,numberOfLossFuncs): #does weights for all loss functions 
  weightsArray=[]
  try:
  	for number in range (0, numberOfLossFuncs):
    		weightsArray.append(epochWeightSineForOneWeight(epoch,numberOfLossFuns,number))
  except:
  	return [1]
  return oneMaker(weightsArray)

#curve fitting fancy

def squareButPreserveSigns(x): 
  print(x)
  if x < 0:
    return -1*x**2
  else:
      return x**2


def squarerDiffSignPreserver(x1,x2): #fix
  print("whqdjadsjhjsahd")
  diff=np.subtract(x1,x2)
  out=[]
  try:
    print("diff", diff)
    for z in range (0,len(diff)):
      out=np.append(out,squareButPreserveSigns(diff[z]))
  except: 
       print("diff is in else ",diff, " ", type(diff))
       return squareButPreserveSigns(diff)
  return out
    

def adaptiveLossNoSin(lossList,weightList):
        diff=lossList[-1]-lossList[-2]
        squaredDiff=squarerDiffSignPreserver(weightList[-1],weightList[-2])
        unit=unitVec(squaredDiff)
        if diff < 0:
                return -1*unit
        return unit

#If weights are improving from before, 
#Generally, we should increase bigger weights more and smaller weights less as it seems bigger weights are more important
#square is to make sure it’s not linear, and for other reasons later explained
#This makes it like you are speeding up or slowing down, sine function fluctuation used before
#square also makes the speed up of individual factors more different, to explore different "offsets" or "phases" better
#does reverse direction if zero or negative, one 
#normalization keeps it bounded
#unitVector function was found to be faster than using a pure NumPy implementation
#This new loss function is cleaner, and is based on differences between weights with sign being preserved
#this makes it grow along derivative of function or negative derivative, so if things looking good with decreasing values for some
#it decreases, if things look good with increasing it increases
#and if it looked bad it flips direction



def curveFancy(lossList, weightList,numberOfLossFuncs):
  minTodoFancy=numberOfLossFuncs+2
  epoch = 1
  if notListOrArray(lossList): epoch=len(lossList)
  if epoch > minTodoFancy: #make sure enough points for curve fitting, else use sine function above to collect more varied points
    newWeights=np.add(adaptiveLossNoSin(lossList,weightList),epochWeightSineBased(epoch,numberOfLossFuncs))
    newWeights[newWeights == 0] = 0.1 #make sure no weight is zero, as then no data there and it vanishes
    print("debug newWeights line 109, ", newWeights, "daptiveLossNoSin(lossList,weightList)",epochWeightSineBased(epoch,numberOfLossFuncs))
    min=np.min(newWeights)
    if min < 0: newWeights=newWeights+min #no negative weights, as this would make things worse in some direction
    return oneMaker(newWeights) #normalized, so it's bounded and nothing grows to extreme
  else:
    return oneMaker(epochWeightSineBased(epoch,numberOfLossFuncs))










#tests to see if functions work
import random


for x in range (-10,10):
	y=random.randint(-100,100)
	z=random.randint(-100,100)
	print("KineticComponet(xVel, yVel): ",y, " ", z," ", KineticComponet(y, z))	
	print("unitVec([x,y]): ",x, " ", z," ", unitVec([x,y]))

for x in range(0,10):
	print("")

for x in range (-10,10):
	numbs=np.random.randint(100, size=60)-50
	print("/n \n numbs are; ,", numbs[0:6]) 
	ans=magneticPotential(numbs[0],numbs[1],numbs[2],numbs[3],numbs[4],numbs[5])
	print(" magneticPotential(magnField,yPos,xPos,xVel,yVel,charge)",  ans)
	print("ans2 ",numbs[6:12])
	ans2=physicLoss(numbs[0],numbs[5],numbs[1],numbs[2],numbs[3],numbs[4],numbs[6],numbs[7],numbs[8],numbs[9])
	print("physicLoss(magField, charge,xPosOrig, yPosnOrig,  xVelOrig, yVelOrig, xPosNow, yPosNow,  xVelNow, yVelNow)")
	print(ans2)

for x in range (-10,10):
	print("")
	numbs=np.random.randint(1,100, size=60)
	print("numbs", numbs[0:3])
	print("epochWeightSineForOneWeight(epoch,numberOfLossFuncs,number)")
	print(epochWeightSineForOneWeight(numbs[0],numbs[1],numbs[2]))

for z in range (-10,10):
	print("")
	numbs=np.random.randint(1,100, size=60)
	print("numbs", numbs[0:2])
	print(epochWeightSineBased(numbs[0],numbs[1]))

def adaptiveLossNoSinTesting(lossList,weightList):
	print("adaptiveLossNoSin(lossList,weightList): ", lossList, " ",weightList)
	print(adaptiveLossNoSin(lossList,weightList))

adaptiveLossNoSinTesting([1, 1, 1, 1, 1],[1,1,1,1,1])
adaptiveLossNoSinTesting([1, 1, 0, 1, 0.1],[1,1,1,1,1])
adaptiveLossNoSinTesting([1, 1, 0, 1, 0.1],[[1,2],[1,2],[1,2],[1,2],[1,2]])
adaptiveLossNoSinTesting([1, 1, 1, 1, 1],[[1,2],[1,2],[1,2],[1,2],[1,2]])


def curveFancyTesting(loss, weightsList, numberOfLossFuncs):
	print(" ")
	print("curveFancy(loss, weightsList, numberOfLossFuncs)")
	print(loss)
	print(weightsList)
	print(numberOfLossFuncs)
	print(curveFancy(loss, weightsList, numberOfLossFuncs))



curveFancyTesting([1, 1, 1, 1, 1],[1,1,1,1,1],1)
curveFancyTesting([1, 1, 0, 1, 0.1],[1,1,1,1,1],1)
curveFancyTesting([1, 1, 0, 1, 0.1],[[1,2],[1,2],[1,2],[1,2],[1,2]],2)
curveFancyTesting([1, 1, 1, 1, 1],[[1,2],[1,2],[1,2],[1,2],[1,2]],2)
curveFancyTesting([1, 1],[[1,2],[1,2]],2)

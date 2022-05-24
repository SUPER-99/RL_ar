import torch

def gradientUpdate(g,PI):
    totalPI = sum(PI)
    weights = []
    for i in range(len(PI)):
        weights.append(PI[i]/totalPI)
    avgG = 0
    for j in range(weights):
        avgG += weights[j]*g[j]
    return avgG

def modelUpdate(wg,PIg,wn,PIn):
    if PIn>PIg:
        wg = wn
        PIg = PIn
    return wg,PIg
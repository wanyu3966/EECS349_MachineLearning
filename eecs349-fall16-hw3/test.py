import numpy as np

x = np.array([1 ,2 ,3])
newX = []
for i in range(len(x)): 
    newX.insert(i, [1.0, x[i]])
print newX

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl 

np.random.seed(937)
data = np.random.lognormal(size=(37, 4), mean=1.5, sigma=1.75)
labels = list('ABCD')
plt.boxplot(data)
plt.show()


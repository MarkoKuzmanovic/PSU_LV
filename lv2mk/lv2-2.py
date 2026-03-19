import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)
mpg = data[:, 0]
hp = data[:, 3]
cyl = data[:, 1]
mpgcyl = np.array([])
hpcyl = np.array([])
dotsize=data[:, 5] * 10
plt.scatter(hp, mpg, marker='*', s=dotsize)
print(mpg.min())
print(mpg.max())
print(mpg.mean())
mpg_6cyl = mpg[cyl == 6]
print(mpg_6cyl.min())
print(mpg_6cyl.max())
print(mpg_6cyl.mean())

plt.show()

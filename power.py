import scipy.stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

filename = 'youtube.graph.small.csv'
df = pd.read_csv(filename)
df = df.drop(df.columns[0],axis=1)
x = np.array(df[df.columns[0]])
y = np.array(df[df.columns[1]])
f, ax = plt.subplots(figsize=(5, 5))


ax.scatter(x, y)
ax.set_title('Distribution')
ax.set_xlabel(df.columns[0])
ax.set_ylabel(df.columns[1])

f, ax = plt.subplots(figsize=(5, 5))
ax.set(xscale="log", yscale="log")
ax.scatter(x, y)
ax.set_title('Log-Log Distribution')
ax.set_xlabel(df.columns[0])
ax.set_ylabel(df.columns[1])

coefficients, residuals, _, _, _ = np.polyfit(np.log(x), np.log(y), 1, full=True)
print("Coefficients",coefficients)
print("Residuals",residuals)
polynomial = np.poly1d(coefficients)
xp = np.linspace(0, 4, 100)
plt.plot(np.log(x), np.log(y), '.', xp, polynomial(xp), '-')

params = scipy.stats.powerlaw.fit(np.log(y))
#Applying the Kolmogorov-Smirnof one sided test
D, p = scipy.stats.kstest(np.log(y), "powerlaw", args=params)
print("p-value: ",str(p),"D: ",str(D))

params = scipy.stats.powerlaw.fit(y)
#Applying the Kolmogorov-Smirnof one sided test
D, p = scipy.stats.kstest(y, "powerlaw", args=params)
print("p-value: ",str(p),"D: ",str(D))
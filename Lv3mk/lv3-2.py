import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('mtcars.csv')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 10))
cilindri = np.unique(df.cyl.values)
meanMPG = [np.mean(df.mpg.values[df.cyl.values == c]) for c in cilindri]
ax1.bar(cilindri.astype(str), meanMPG)
ax1.set_xlabel('CYL')
ax1.set_ylabel('MPG')


wt = [df.wt.values[df.cyl.values == c] for c in cilindri]
ax2.boxplot(wt, tick_labels=[f"{c} cil" for c in cilindri])
ax2.set_xlabel('CYL')
ax2.set_ylabel('WT')

auto = df.mpg.values[df.am.values == 0]
manual = df.mpg.values[df.am.values == 1]
ax3.boxplot([auto, manual], tick_labels=['Automatski (0)', 'Rucni (1)'])
ax3.set_title('AM')
ax3.set_ylabel('MPG')

hp = df.hp.values
acc = df.qsec.values * 0.25
am = df.am.values
ax4.scatter(hp[am == 0], acc[am == 0])
ax4.scatter(hp[am == 1], acc[am == 1])
ax4.set_xlabel('HP')
ax4.set_ylabel('Time')
ax4.legend()


plt.tight_layout()
plt.show()
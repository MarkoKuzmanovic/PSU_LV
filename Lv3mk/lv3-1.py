import pandas as pd
import numpy as np
mtcars = pd.read_csv('mtcars.csv')
print(mtcars.sort_values(by=['mpg']).tail(5))
print('---------')
print(mtcars[mtcars.cyl == 8].sort_values(by=['mpg']).head(3))
print('---------')
print(mtcars[mtcars.cyl == 6].mpg.mean())
print('---------')
print(mtcars[(mtcars.cyl == 4) & (mtcars.wt >= 2) & (mtcars.wt <= 2.2)].mpg.mean())
print('---------')
print(mtcars[(mtcars.am == 0 )].am.count())
print(mtcars[mtcars.am == 1].am.count())
print('---------')
print(mtcars[(mtcars.am == 0) & (mtcars.hp > 100)].car.count())
print('---------')
mtcars['wt_kg'] = mtcars['wt'] * 0.45359237
print(mtcars[['car', 'wt_kg']])


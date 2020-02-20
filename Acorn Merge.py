import pandas as pd
import numpy as np
from pandas import datetime
from matplotlib import pyplot as plt
import os
from matplotlib import pyplot

mac000216 = ['193','99','63','50']
mac000046 = ['160','88','85','76']
mac000213 = ['98','99','98','114']

df = pd.read_csv("C:/Users/A02290684/Desktop/clean energy/Project/.idea/Acorn Csv/MAC000213.csv")
df = df[['day','LCLid','energy_sum']]

df['Acorn_value1'] = 98
df['Acorn_value2'] = 99
df['Acorn_value3'] = 98
df['Acorn_value4'] = 114

df.to_csv("C:/Users/A02290684/Desktop/clean energy/Project/.idea/Acorn Csv/MAC000213_With_Acorn.csv")

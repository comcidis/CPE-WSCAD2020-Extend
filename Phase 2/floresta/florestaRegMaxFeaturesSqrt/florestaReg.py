#random_florest_reg_base_Real

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys

#------------ sys.argv = ARGUMENTOS QUE VÃO VIR DIRETO DA EXECUÇÃO DO PYTHON3 ----------#
# Pode vir um valor "Default", que é o algoritmo default para o sklearn. por isso precisa saber antes de inserir o parâmetro.

try:
    maxDepth = int(sys.argv[1])
except:
    maxDepth = None
try:
    nEstimators = int(sys.argv[2])
except:
    nEstimators = 100
try:
    nJobs = int(sys.argv[3])
except:
    nJobs = 1

df = pd.read_csv("../For_modeling-5milhoes.csv")

x = df.iloc[:,1:25]
y = df.iloc[:,:1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

arvore = RandomForestRegressor(max_features='sqrt',max_depth=maxDepth,n_estimators=nEstimators,n_jobs=nJobs,random_state=0)
arvoreMontada = arvore.fit(x_train,y_train.values.ravel())

previsao = np.array(arvore.predict(x_test.values))

mean_Squared_error = mean_squared_error(y_test,previsao)

print(f'mean_squared_error:{mean_Squared_error}')

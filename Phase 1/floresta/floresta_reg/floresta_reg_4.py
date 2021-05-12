#arvore_regressora_base_numerica_e_nominal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import sys

#------------ ARGUMENTOS QUE VÃO VIR DIRETO DA EXECUÇÃO DO PYTHON3 ----------#
numeroExemplos = int(sys.argv[1])
numeroAtributo = int(sys.argv[2])

encoder = OneHotEncoder()

df = pd.read_csv("../../../../BaseNova/"+str(numeroExemplos)+"kk_"+str(numeroAtributo)+"att.csv",sep=",")

df2 = df.loc[:, df.columns != 'class']
df3 = pd.get_dummies(df2)

x = df3.iloc[:,0:len(df3.columns)]

yCategorico = df.iloc[:,-1:]
y = encoder.fit_transform(yCategorico).toarray()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

floresta = RandomForestRegressor(random_state=0)
florestaMontada = floresta.fit(x_train,y_train)

previsao = np.array(floresta.predict(x_test.values))

mean_Squared_error = mean_squared_error(y_test,previsao)

print(f'mean_squared_error:{mean_Squared_error}')

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import sys

#------------ ARGUMENTOS QUE VÃO VIR DIRETO DA EXECUÇÃO DO PYTHON3 ----------#
numeroExemplos = int(sys.argv[1])
numeroAtributo = int(sys.argv[2])

encoder = OneHotEncoder()

df = pd.read_csv("../../../../BaseSintetica/"+str(numeroExemplos)+"k_"+str(numeroAtributo)+"att_categ.csv",sep=",")

xCategorico = df.iloc[:,0:numeroAtributo]
yCategorico = df.iloc[:,-1:]

x = encoder.fit_transform(xCategorico).toarray()
y = encoder.fit_transform(yCategorico).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

arvore = DecisionTreeRegressor(random_state=0)
arvoreMontada = arvore.fit(x_train,y_train)

previsao = np.array(arvore.predict(x_test))

mean_Squared_error = mean_squared_error(y_test,previsao)

print(f'mean_squared_error:{mean_Squared_error}')


#arvore_classificadora_base_numerica_e_nominal

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

le = preprocessing.LabelEncoder()

#------------ ARGUMENTOS QUE VÃO VIR DIRETO DA EXECUÇÃO DO PYTHON3 ----------#
numeroExemplos = int(sys.argv[1])
numeroAtributo = int(sys.argv[2])

df = pd.read_csv("../../../../BaseNova/"+str(numeroExemplos)+"kk_"+str(numeroAtributo)+"att.csv",sep=",")

df2 = df.loc[:, df.columns != 'class']
df3 = pd.get_dummies(df2)

x = df3.iloc[:,0:len(df3.columns)]
y = df.iloc[:,-1:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

arvore = DecisionTreeClassifier(random_state=0)
arvoreMontada = arvore.fit(x_train,y_train)

previsao = np.array(arvore.predict(x_test.values))
acuracia = accuracy_score(y_test, previsao)

print(f'acuracia:{acuracia}')


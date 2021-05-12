import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

#------------ ARGUMENTOS QUE VÃO VIR DIRETO DA EXECUÇÃO DO PYTHON3 ----------#
numeroExemplos = int(sys.argv[1])
numeroAtributo = int(sys.argv[2])

df = pd.read_csv("../../../../BaseSintetica/"+str(numeroExemplos)+"k_"+str(numeroAtributo)+"att_num.csv",sep=",",index_col=False)

x = df.iloc[:,0:numeroAtributo]
y = df.iloc[:,-1:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier(random_state = 0)
florestaMontada = rf.fit(x_train,y_train.values.ravel())

previsao = np.array(rf.predict(x_test.values))

acuracia = accuracy_score(y_test, previsao)

print(f'acuracia:{acuracia}')



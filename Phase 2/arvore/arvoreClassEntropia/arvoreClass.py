#Arvore_class_base_Real

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys

#------------ sys.argv = ARGUMENTOS QUE VÃO VIR DIRETO DA EXECUÇÃO DO PYTHON3 ----------#
# Pode vir um valor "Default", que é o algoritmo default para o sklearn. por isso precisa saber antes de inserir o parâmetro.

try:
    maxDepth = int(sys.argv[1])
except:
    maxDepth = None

try:
    minSampleSplit = int(sys.argv[2])
except:
    minSampleSplit = 2

df = pd.read_csv("../HIGGS-5milhoes-28att.csv")

x = df.iloc[:,1:29]
y = df.iloc[:,:1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

arvore = DecisionTreeClassifier(criterion='entropy',max_depth=maxDepth,min_samples_split=minSampleSplit,random_state=0)
arvoreMontada = arvore.fit(x_train,y_train)

previsao = np.array(arvore.predict(x_test.values))

target_names = ['class 1', 'class 0']
print(classification_report(y_test, previsao, target_names=target_names))



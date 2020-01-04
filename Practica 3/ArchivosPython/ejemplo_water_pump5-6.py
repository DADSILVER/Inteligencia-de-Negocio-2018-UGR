# -*- coding: utf-8 -*-
"""
Autor:
    David Castro Salazar
Fecha:
    Noviembre/2018
Contenido:
    Uso simple de XGB y LightGBM para competir en DrivenData:
       https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

le = preprocessing.LabelEncoder()

'''
lectura de datos
'''
#los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
data_x = pd.read_csv('water_pump_tra.csv')
data_y = pd.read_csv('water_pump_tra_target.csv')
data_x_tst = pd.read_csv('water_pump_tst.csv')

#se quitan las columnas que no se usan
data_x.drop(labels=['id'], axis=1,inplace = True)
data_x.drop(labels=['amount_tsh'], axis=1,inplace = True)
data_x.drop(labels=['num_private'], axis=1,inplace = True)
#data_x.drop(labels=['district_code'], axis=1,inplace = True)
data_x.drop(labels=['wpt_name'], axis=1,inplace = True)
data_x.drop(labels=['region'], axis=1,inplace = True)
data_x.drop(labels=['recorded_by'], axis=1,inplace = True)
data_x.drop(labels=['scheme_name'], axis=1,inplace = True)
data_x.drop(labels=['extraction_type_group'], axis=1,inplace = True)
data_x.drop(labels=['extraction_type_class'], axis=1,inplace = True)
data_x.drop(labels=['management_group'], axis=1,inplace = True)
data_x.drop(labels=['payment_type'], axis=1,inplace = True)
data_x.drop(labels=['water_quality'], axis=1,inplace = True)
data_x.drop(labels=['quantity_group'], axis=1,inplace = True)
data_x.drop(labels=['source_class'], axis=1,inplace = True)
#data_x.drop(labels=['source_type'], axis=1,inplace = True)
data_x.drop(labels=['waterpoint_type_group'], axis=1,inplace = True)



data_x_tst.drop(labels=['id'], axis=1,inplace = True)
data_x_tst.drop(labels=['amount_tsh'], axis=1,inplace = True)
data_x_tst.drop(labels=['num_private'], axis=1,inplace = True)
#data_x_tst.drop(labels=['district_code'], axis=1,inplace = True)
data_x_tst.drop(labels=['wpt_name'], axis=1,inplace = True)
data_x_tst.drop(labels=['region'], axis=1,inplace = True)
data_x_tst.drop(labels=['recorded_by'], axis=1,inplace = True)
data_x_tst.drop(labels=['scheme_name'], axis=1,inplace = True)
data_x_tst.drop(labels=['extraction_type_group'], axis=1,inplace = True)
data_x_tst.drop(labels=['extraction_type_class'], axis=1,inplace = True)
data_x_tst.drop(labels=['management_group'], axis=1,inplace = True)
data_x_tst.drop(labels=['payment_type'], axis=1,inplace = True)
data_x_tst.drop(labels=['water_quality'], axis=1,inplace = True)
data_x_tst.drop(labels=['quantity_group'], axis=1,inplace = True)
data_x_tst.drop(labels=['source_class'], axis=1,inplace = True)
#data_x_tst.drop(labels=['source_type'], axis=1,inplace = True)
data_x_tst.drop(labels=['waterpoint_type_group'], axis=1,inplace = True)



data_y.drop(labels=['id'], axis=1,inplace = True)



'''
Dividimos data_recored en dia/mes/año y quitamos la columna de dia
'''


dfmes=[]
dfAnno=[]

dfmes_tst=[]
dfAnno_tst=[]

for var in range(0,data_x['date_recorded'].size):
    dfmes.append(data_x['date_recorded'][var][5:7])
    dfAnno.append(data_x['date_recorded'][var][0:4])
    
for var in range(0,data_x_tst['date_recorded'].size):    
    dfmes_tst.append(data_x_tst['date_recorded'][var][5:7])
    dfAnno_tst.append(data_x_tst['date_recorded'][var][0:4])

# Añadimos las dos nuevas columnas a cada dataframe

data_x['Mes']=dfmes
data_x['anno']=dfAnno

data_x_tst['Mes']=dfmes_tst
data_x_tst['anno']=dfAnno_tst

#borramos la columna de date_recorded
data_x.drop(labels=['date_recorded'], axis=1,inplace = True)
data_x_tst.drop(labels=['date_recorded'], axis=1,inplace = True)


def changeNaNMean(columna, valor):
    data_x[columna].replace(valor,np.nan,inplace=True)
    data_x[columna].fillna(round(data_x.groupby(['region_code'])[columna].transform("mean")), inplace=True)
    data_x[columna].fillna(round(data_x[columna].mean()), inplace=True)
    data_x_tst[columna].replace(valor,np.nan,inplace=True)
    data_x_tst[columna].fillna(round(data_x_tst.groupby(['region_code'])[columna].transform("mean")), inplace=True)
    data_x_tst[columna].fillna(round(data_x_tst[columna].mean()), inplace=True)
    


def changeNaNMode(columna1):
    data_x[columna1].fillna(data_x[columna1].mode()[0], inplace=True)
    data_x_tst[columna1].fillna(data_x_tst[columna1].mode()[0], inplace=True)


'''
Ajuste de variables
'''

changeNaNMean("gps_height",0.0)
changeNaNMean("population",0.0)
changeNaNMean("construction_year",0.0)


changeNaNMode("subvillage")
changeNaNMode("scheme_management")
changeNaNMode("permit")
changeNaNMode("funder")
changeNaNMode("installer")
changeNaNMode("public_meeting")


dfTiemFunc=[]
dfTiemFunc_tst=[]
for var in range(0,data_x['construction_year'].size):    
    dfTiemFunc.append(int(data_x['anno'][var])-data_x['construction_year'][var])
    
for var in range(0,data_x_tst['construction_year'].size):    
    dfTiemFunc_tst.append(int(data_x_tst['anno'][var])-data_x_tst['construction_year'][var])

data_x['Tiempo_func']=dfTiemFunc
data_x_tst['Tiempo_func']=dfTiemFunc_tst



'''
Fin de ajuste de variables
'''

'''
Se convierten las variables categóricas a variables numéricas (ordinales)
'''
from sklearn.preprocessing import LabelEncoder
mask = data_x.isnull()
data_x_tmp = data_x.fillna(9999)
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform)
data_x_nan = data_x_tmp.where(~mask, data_x)

mask = data_x_tst.isnull() #máscara para luego recuperar los NaN
data_x_tmp = data_x_tst.fillna(9999) #LabelEncoder no funciona con NaN, se asigna un valor no usado
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform) #se convierten categóricas en numéricas
data_x_tst_nan = data_x_tmp.where(~mask, data_x_tst) #se recuperan los NaN




X = data_x_nan.values
X_tst = data_x_tst_nan.values
y = np.ravel(data_y.values)

#------------------------------------------------------------------------


'''
Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''

#submission5
skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=2147483647)

'''
#submission6
skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=2147483647)
'''
le = preprocessing.LabelEncoder()

def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []

    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("Score: {:.4f}, tiempo: {:6.2f} segundos".format(accuracy_score(y[test],y_pred) , tiempo))
        y_test_all = np.concatenate([y_test_all,y[test]])

    print("")

    return modelo, y_test_all
#------------------------------------------------------------------------



### Submission4
print("------ RandomForest...")
clf = RandomForestClassifier(n_estimators=1000,n_jobs = -1,random_state=2147483647)

rfvc, y_prob_lgbm = validacion_cruzada(clf,X,y,skf)
#lgbm, y_test_lgbm, y_prob_lgbm = validacion_cruzada(lgbm,X,y,skf)

#clf = clf.fit(X,y)

y_pred_tra = rfvc.predict(X)
print("Score: {:.4f}".format(accuracy_score(y,y_pred_tra)))
y_pred_tst = rfvc.predict(X_tst)




df_submission = pd.read_csv('water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission1000.csv", index=False)
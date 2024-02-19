# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:03:04 2023

@author: jcgarciam
"""

import bz2
import pandas as pd


path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo)
    
#%%

print(df.groupby('DayOfWeek')['ArrDelay'].describe())

#%%

print(df.groupby('DayOfWeek')['ArrDelay','DepDelay'].mean())

#%%

print(df.groupby(['Origin','DayOfWeek'])['ArrDelay'].max())

#%%

dfduplicate = df.append(df)

#dfduplicate = dfduplicate.sample(frac = 1)

dfclean = dfduplicate.drop_duplicates()

#%%

import numpy as np

valoraciones = np.array([[8,7,8,5],[2,6,8,1],[8,8,9,5]])

print(valoraciones)

#%%

valoraciones2 = np.array([[[8,7,8,5],[2,5,5,2]],[[2,6,8,4],[8,9,7,4]],[[8,8,9,3],[10,9,10,8]]])

print(valoraciones2)

print(valoraciones2[0,0,0])

#%%

print(np.zeros((3,2,4)))

#%%

print(valoraciones2)
print(valoraciones2 + np.ones((3,2,4)))

#%%

print(np.mean(valoraciones2))

#%%

print(np.reshape([1,2,3,4,5,6,7,8,9,10,11,12], (3,2,2)))

#%%

print(df['ArrTime'].mean())
print(np.mean(df['ArrTime']))

#%%

df.dropna(inplace = True, subset = ['ArrDelay','DepDelay'])

print(np.corrcoef(df['ArrDelay'], df['DepDelay']))

#%%


##### Test de chi cuadrado
import bz2
import pandas as pd


path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo)

import numpy as np

np.random.seed(0) #fijamos una semilla para obtener siempre los mismos resultados

df = df[df['Origin'].isin(['HOU','ATL','IND']) == True]
df = df.sample(frac = 1) #reordenamos las filas
df = df[0:10000] #tomamos las primeras 10mil filas ya que no es necesario trabajar con todas

#%%

df['BigDelay'] = df['ArrDelay'] > 30

# Creamos una tabla de contigencia, es una tabla que agrupa por frecuencias
observados = pd.crosstab(index = df['BigDelay'], columns = df['Origin'], margins = True)
print(observados)

#%%

from scipy.stats import chi2_contingency

#%%

test = chi2_contingency(observados)

print(test) #el primer dato es el estadistico (la suma de las diferencias al cuadrado)
#el segundo es el p-value, el cuarto es una tabla de valores esperados

#%%

#convertimos la tabla de valores esperados en un dataframe, estos son los valores 
#teoricos esperados si no hubiese ningun tipo de relacion

esperados = pd.DataFrame(test[3])
print(esperados)

#%%

# creamos una tabla de valores relativos de los valores esperados y una 
# de los valores observados para que al compararlos sea mucho mas facil

esperados_rel = round(esperados.apply(lambda r:r/len(df)*100, axis = 1), 2)
observados_rel = round(observados.apply(lambda r:r/len(df)*100, axis = 1), 2)

print(observados_rel)
print(esperados_rel)
#con estas tablas podemos comparar dato a dato si uno es sufieciente mayor
#o poco. Necesitamos una medida que compare todos los datos
#
#%%
#si p-value <0.05, hay diferencias significativas. Hay relación entre las variables
# sino, no hay diferencias significativas. No hay relacion entre variables
print(test[1]) # no se puede afirmar que haya relación entre variables.

#%%
# Analisis de datos extremos
with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo, nrows = 100000)
    
#%%

x = df['ArrDelay'].dropna()

q1 = np.percentile(x, 25)
q3 = np.percentile(x, 75)
rangointer = q3-q1

#%%

umbralsuperior = q3 + 1.5*rangointer
umbralinferior = q1 - 1.5*rangointer
#%%
# cualquier valor que este por encima de este valor va a ser un valor atipico
print(umbralsuperior)
# cualquier valor que este por debajo de este valor va a ser un valor atipico
print(umbralinferior)

#%%

print(round(np.mean(x > umbralsuperior)*100,2)) # el 8.39% de los casos esta por encima
print(round(np.mean(x < umbralinferior)*100,2)) #menos del 2% de los casos esta por debajo
#estas medidas no son simetricas, nos da un umbral de los datos que estan por
#encima o por debajo pero no nos asegura que sea igual

#%%
#podemos estudiar varias variables a la vez
from sklearn.covariance import EllipticEnvelope

#%%
#creamos un modelo que tome el 1% de los datos mas alejados
outliers = EllipticEnvelope(contamination = .01) 

#%%

var_lista = ['DepDelay','TaxiIn','TaxiOut','CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay']

x = np.array(df.loc[:,var_lista].dropna())

#%%
#entrenamos nuestro modelo con los datos

outliers.fit(x)

#%%

# le pedimos una prediccion de que valores forman parte del 1% que queremos detectar

pred = outliers.predict(x) # es un array de unos y menos unos, y lo que 
# nos va a interesar son los menos unos
print(pred)

#%%
elips_outliers = np.where(pred == -1)[0]

print(elips_outliers) #datos extremadamente alejados


#%%

# Paralelizar loops. Cuando paralelizamos un proceso le pedimos al codigo que use
# uno o mas nucleos de los que tenemos

import bz2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo)
    

#%%

df_sub = df[['CarrierDelay','WeatherDelay','NASDelay', 'SecurityDelay', 'LateAircraftDelay']]
print(df_sub.head(10))


#%%
# funcion para encontrar el retraso maximo por fila
def retraso_maximo(fila):
    if not np.isnan(fila).any():
        print(np.isnan(fila).any())
        names = ['CarrierDelay','WeatherDelay','NASDelay', 'SecurityDelay', 'LateAircraftDelay']
        return names[fila.index(max(fila))]
    
    else:
        return 'None'
    
#%%

results = []

for fila in df_sub.values.tolist():
    results.append(retraso_maximo(fila))

print(results)
        
        
#%%
# paralelizamos el proceso
result = Parallel(n_jobs = 2, backend = 'multiprocessing')( #numero de procesadores 2
            map(delayed(retraso_maximo), df_sub.values.tolist()))
result


#%%

#INTRODUCCION MATPLOTLIB

import bz2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo)
    
#%%

data = np.unique(df.Cancelled, return_counts = True)
print(data)

#%%

plt.pie(x = data[1], labels = data[0])    
plt.show()
        
#%%   

plt.pie(x = data[1], 
        labels = data[0], 
        colors = ['Red','Green'], 
        shadow = True, 
        radius = 2)    
plt.show()  
        
#%%
#MODIFICAR EL ELEMENTO DEL GRAFICO EN MATPLOTLIB


import bz2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo)
    
#%%

df = df.sample(frac = 1).head(100)

#%%

plt.scatter(x = df.DayofMonth, y = df.ArrDelay, s = df.Distance)

#%%

plt.scatter(x = df.DayofMonth, y = df.ArrDelay, s = df.Distance, alpha = 0.3, c = df.DayOfWeek.isin([6,7]))
plt.title('Retrasos en EEUU')      
plt.xlabel('Día del mes')
plt.ylabel('Retraso al llegar')
plt.ylim([0,150])
plt.xticks([0,15,30])
plt.text(x = 28, y = 120, s = 'Mi Vuelo')

#%%

## ETIQUETAS Y LEYENDAS
import bz2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo, nrows = 100000)

#%%

data = np.unique(df.DayOfWeek, return_counts = True)
labels = ['Lun','Mar','Mie','Jue','Vie','Sab','Dom']
   
#%%

plt.pie(x = data[1],
        labels = data[0],
        radius = 1.5,
        colors = ['Red','Green','Orange','Blue','Gray','Pink','Black'],
        startangle = 90)

#%%

plt.pie(x = data[1],
        #labels = labels,
        radius = 0.7,
        colors = sns.color_palette('hls',7),
        explode = (0,0,0,0,0,0,0.1),
        startangle = 90,
        autopct = '%1.1f%%',
        labeldistance = 1)
plt.legend(loc = 'upper left', labels = labels)

#%%

plt = sns.barplot(x = labels, y = data[1])
plt.set(xlabel = 'Dia de la semana', ylabel = 'Numero de vuelos')

#%%
plt = sns.pieplot(x = labels, y = data[1])
plt.set(xlabel = 'Dia de la semana', ylabel = 'Numero de vuelos')

#%%

# GRAFICOS PARA SERIES TEMPORALES

import bz2
import numpy as np
import pandas as pd
import seaborn as sns
import datetime

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo)

df2 = df[df['Origin'].isin(['ATL','HOU','IND'])]
df = df.head(500000)

#%%

times = []

for i in np.arange(len(df)):
    times.append(datetime.datetime(year = 2008, month = df.loc[i,'Month'], day = df.loc[i, 'DayofMonth']))

#%%

df['Time'] = times

data = df.groupby(by = 'Time', as_index = False)['DepDelay','ArrDelay'].mean()

#%%

sns.lineplot(data['Time'], data['DepDelay'])
        
#%%

data = df.groupby(by = 'Time')['DepDelay','ArrDelay'].mean()
                  
sns.lineplot(data = data)

#%%

times = []

for i in df2.index:
    times.append(datetime.datetime(year = 2008, month = df2.loc[i,'Month'], day = df2.loc[i, 'DayofMonth']))

df2['Time'] = times

#%%

sns.set(rc = {'figure.figsize':(15,10)})
sns.lineplot(x = 'Time', y = 'ArrDelay', hue = 'Origin', data = df2)

#%%

#### HISTOGRAMAS Y BOX PLOTS EN MATPLOTLIB

import bz2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo)

#%%

df.dropna(inplace = True, subset = ['ArrDelay','DepDelay','Distance'])

#%%

sns.distplot(df['Distance'], kde = False, bins = 100) #distribucion de los datos

#%%
# podemos comparar dos distribuciones
sns.kdeplot(df['ArrDelay'])
sns.kdeplot(df['DepDelay'])
plt.xlim([-300,300])

#%%

df2 = df[df['Origin'].isin(['ATL','HOU','IND'])].sample(frac = 1).head(500)

#%%

sns.boxplot(x = 'DepDelay', y = 'Origin', data = df2)
plt.xlim([-20,150])

#%%

#### NUBES DE PUNTOS Y MAPAS DE CALOR EN MATPLOTLIB


import bz2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo)

#%%

df.dropna(inplace = True, subset = ['ArrDelay','DepDelay','Distance','AirTime'])

#%%

sns.set(rc = {'figure.figsize':(15,10)})

#%%
# grafica que nos relaciona dos variables y le añade un histograma a cada variable
df2 = df[df['Origin'].isin(['ATL','HOU','IND'])].sample(frac = 1).head(1000)

sns.jointplot(df2['DepDelay'], df2['ArrDelay'])

#%%
df3 = df2[np.abs(df2['DepDelay']) < 40]
df3 = df3[np.abs(df2['ArrDelay']) < 40]

#%%
#losgonos nos muestran donde estan mas concentrados los datos
sns.jointplot(df3['DepDelay'], df3['ArrDelay'], kind = 'hex')

#%%
# esta grafica en vez de histogramas nos muestra plot de densidad y en vez
# de hexagonos curvas d enivel que nos deja ver mejor la concenctracion de los datos
sns.jointplot(df3['DepDelay'], df3['ArrDelay'], kind = 'kde')

#%%

gb_df = pd.DataFrame(df2.groupby(['Origin','Month'], as_index = False)['DepDelay'].mean())

data = gb_df.pivot('Month','Origin','DepDelay')
print(data)

#%%

sns.heatmap(data = data, annot = True, linewidths = .5)


#%%

# NECESIDADES DE MACHINE LEARNING: CLUSTERING Y MODELIZACION

"""
Aprendizaje supervisado:
    - Modelizacion de datos
    - Informacion sobre la variable de interes
    - Explicacion/Relacion
    
Aprendizaje no supervisado:
    -clustering
    -sin informacion sobre la variable de interes
    -Generacion de informacion
"""

#%%

# PREPARAR LOS DATOS PARA MACHINE LEARNING

import bz2
import numpy as np
import pandas as pd
from sklearn import preprocessing

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo, nrows = 100000)

df = df[['ArrDelay','DepDelay','Distance','AirTime']].dropna()

#%%
#algunos de los metodos de machine learning necesitan que los datos esten escalados.
# la mas popular es la estandarizacion. Este metodo transforma los datos para que tengan
# una escala igual a cero y una desviacion igual a 1.
# z = (x-media)/desviacion_estandar, formula que se aplica por defecto para cada columna
# usando la funcion scale 

X_scaled = preprocessing.scale(df)
print(X_scaled)

#%%

print(X_scaled.mean(axis = 0))
print(X_scaled.std(axis = 0))

# La estandarizacion nos permite comparar mas facilmente la variables ya que la 
# media ahora e sigual a cero para todos las variables

print(df.iloc[2])
print(X_scaled[2])

#%%

# podemos aplicar una nueva escala

min_max_scaler = preprocessing.MinMaxScaler([0,10]) # le especificamos que rango de valores queremos que tenga
X_train_minmax = min_max_scaler.fit_transform(df) #transformamos los datos
print(X_train_minmax)
# estas transformaciones son muy buenas en caso de que tengamos muchos outliers
#%%

#tambien podemos transformar variables categoricas en variables dummi


with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo, nrows = 100)
#%%
# ejemplo
print(pd.get_dummies(df['Origin']))
# genera un arreglo de las n filas por la cantidad de categorias que hayan
# llenandolas de unos y ceros. Uno va aparecer en la fila y columna donde si aparezca
# la categoria y de resto ceros.
# Punto a favor permite ver mejor si las variables están relacionadas.
# Por otro lado este proceso concume mas memoria

#%%

# METODO DE K-MEANS. EL ALGORITMO DE CLUSTERING
# este metodo separa los datos en n-grupos, asegurando que cada grupo tenga la misma
# varianza.
# Genera grupos disjuntos, que se consigue agrupando cada punto al centroide mas cercano

import bz2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo, nrows = 1e5)

newdf = df[['AirTime','DepDelay']].dropna()

#%%
#entrenamos el modelo de K-means

kmeans = KMeans(n_clusters = 4, random_state = 0, n_jobs = -1).fit(newdf) #random_state = 0 es para mantener los mismos resultados
print(kmeans.labels_) #muestra los el grupo al que pertenece cada dato

print(np.unique(kmeans.labels_, return_counts = True))

#%%

import matplotlib.pyplot as plt

plt.scatter(newdf['AirTime'], newdf['DepDelay'], c = kmeans.labels_)

#%%

print(kmeans.cluster_centers_) #muestra las coordenadas de los centroides

#%%

#Predecimos en que grupo van a estar nuevos puntos
with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo, nrows = 1e6)
alldf = df[['AirTime','DepDelay']].dropna()

#%%


print(kmeans.predict(alldf)[0:50])

#%%

#podemos clasificar con mas variables

newdf = df[['AirTime','Distance','TaxiOut','ArrDelay','DepDelay']].dropna()

#%%


kmeans = KMeans(n_clusters = 4, random_state = 0, n_jobs = -1).fit(newdf) #random_state = 0 es para mantener los mismos resultados
print(kmeans.labels_) #muestra los el grupo al que pertenece cada dato

print(np.unique(kmeans.labels_, return_counts = True))

#%%

# visualizamos solo dos variables

import matplotlib.pyplot as plt

plt.scatter(newdf['AirTime'], newdf['DepDelay'], c = kmeans.labels_)


#%%

# EL ALGORITMO HIERARCHICAL CLUSTERING

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import bz2

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo, nrows = 1e4)

newdf = df[['AirTime','DepDelay']].dropna()

#%%

clstr = AgglomerativeClustering(n_clusters = 5)
clstr.fit(newdf)

#%%

plt.scatter(newdf['AirTime'], newdf['DepDelay'], c = clstr.fit_predict(newdf))


#%%

#REGRESION LINEAL
# Y = B0 + B1X1 + B2X2 + .... E
# 1. Relacion lineal entre variables, es decir que cuando incrementamos en una
#    x veces en una variable, se incrementa en promedio beta-veces la variable respuesta
# 2. Errores independientes, es decir que los errores entre las variables explicativas, 
#    sean independientes entre si. Tambien se asume que, las variables explicativas no
#    estan relacionadas linealmente entre ellas.
# 3. Homocedasticidad: los errores tengan varianza constante.
# 4. Esperanza(errores) = 0
# 5. Error total = sum(errores_i)

import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import bz2

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo)
    
#%%

df = df.dropna(subset = ['ArrDelay'])
df = df.sample(frac = 1).head(100000)
y = df['ArrDelay'] #retraso en la llegada
x = df[['DepDelay']] #retraso en la salida

#%%

regr = linear_model.LinearRegression()
regr.fit(x,y)

print('Coeficientes: ', regr.coef_)
print('Intercepto: ', regr.intercept_)
y_pred = regr.predict(x)
print('R cuadrado: ', r2_score(y,y_pred))
mse = mean_squared_error(y, y_pred)
print(f'El error cuadratico medio es: {mse:.2f}')
#%%

plt.scatter(x[1:10000], y[1:10000], color = 'black')
plt.plot(x[1:10000], y_pred[1:10000], color = 'blue')
plt.show()

#%%

# hacemos un modelo con variables categoricas

x = df[['AirTime','Distance','TaxiIn','TaxiOut']]

df['Month'] = df['Month'].apply(str)
df['DayofMonth'] = df['DayofMonth'].apply(str)
df['DayOfWeek'] = df['DayOfWeek'].apply(str)

dummies = pd.get_dummies(data = df[['Month','DayofMonth','DayOfWeek','Origin','Dest']])
x = dummies.add(x, fill_value = 0)
#%%

regr = linear_model.LinearRegression()
regr.fit(x,y)
y_pred = regr.predict(x)
print('R cuadrado: ',r2_score(y, y_pred))

#%%

x = x.add(df[['DepDelay']], fill_value = 0)

#%%

regr = linear_model.LinearRegression()
regr.fit(x,y)
y_pred = regr.predict(x)
print('R cuadrado: ',r2_score(y, y_pred))

#%%

# REGRESION LOGISTICA
#
# la regresion lineal tiene una version que permite obtener valores categoricos
# en vez de valores de variables continuas. Esta version modela la probabilidad
# de que obtengamos cada una de las categorias. Tipicamente se conoce como modelo
# binario, por ejemplo para predecir si un cliente compra o no un producto,
# pero el modelo esta adaptado para predecir un numero ilimitado de categorias.


# P(y) = 1/(1+ e^(B0 + B1X1 + B2X2 + .... E)) #probabilidad de cierta categoria

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import bz2

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo)

#%%

df = df.dropna(subset = ['ArrDelay'])
df = df.sample(frac = 1).head(100000)
y = df['ArrDelay'] < 1 
x = df[['DepDelay']]

#%%

logreg = LogisticRegression()
logreg.fit(x,y)
y_pred = logreg.predict(x)

#%%
print(y_pred)
print(np.round(logreg.predict_proba(x), 3))
print(np.mean(y_pred == y)) #evaluamos la eficiencia del comelo comparando los valores reales con las predicciones y obtenemos la media
print(np.mean(y)) # evaluamos si las categoria estan bien compensadas para no soobre estimar el modelo


#%%

# NAIVES BAYES CLASSIFIER

# P(y|x1,...,xn) = P(y)*P(x1,...,xn|y)/P(x1,...,xn)

from sklearn.naive_bayes import BernoulliNB
#from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import bz2

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo)

df = df.sample(frac = 1)
df = df.head(500000)

#%%

df = df.dropna(subset = ['ArrDelay'])

y = df['ArrDelay'] > 0 #vuelos que se han retrasado un minuto

#%%

df['Month'] = df['Month'].apply(str)
df['DayofMonth'] = df['DayofMonth'].apply(str)
df['DayOfWeek'] = df['DayOfWeek'].apply(str)
df['TailNum'] = df['TailNum'].apply(str)

x = pd.get_dummies(data = df[['Month','DayofMonth','DayOfWeek','Origin',
                              'Dest','UniqueCarrier','TailNum']])

#%%
#habitualmente el siguiente modelo se utiliza en analisis de texto
clf = BernoulliNB()
clf.fit(x,y)
y_pred = clf.predict(x)

#%%

print(np.mean(y == y_pred)) #Evaluamos la cantidad de aciertos

print(1 - np.mean(y)) # Se evalua el porcentaje de la media y evaluamos que solo ganamos un porcentaje mas en nuestro modelo predictivo

#%%

x = df[['AirTime','Distance','TaxiIn','TaxiOut']]
clf = GaussianNB()
clf.fit(x,y)
y_pred = clf.predict(x)

#%%
print(np.mean(y == y_pred)) #Evaluamos la cantidad de aciertos

print(1 - np.mean(y))

#%%

# ARBOLES DE CLASIFICACION Y REGRESION

"""
La idea principal detras de los arboles de clasificacion es seleccionar la variable
que sea mas explicativa a la hora de partir todos los casos que disponemos en dos grupos.
"""

from sklearn import tree
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
import bz2

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo)

df = df.dropna(subset = ['ArrDelay'])
df = df.sample(frac = 1)
dftest = df.tail(500000)
df = df.head(500000)


#%%
#entrenamiento de variables categoricas
clf = tree.DecisionTreeClassifier()

x = df[['Distance','AirTime','DepTime','TaxiIn','TaxiOut','DepDelay']]
x_test = dftest[['Distance','AirTime','DepTime','TaxiIn','TaxiOut','DepDelay']]
y = df['ArrDelay'] > 10
y_test = dftest['ArrDelay'] > 10

clf = clf.fit(x,y)
y_pred = clf.predict(x)
y_pred_test = clf.predict(x_test)

#%%

print(np.mean(y == y_pred)) # no tiene mucho sentido que prediga casi el 100%. Esto
#sucede porque el modelo crea un arbol que se ajuste casi perfectamente a los datos,
# por eso se han creado las dos bases de datos df y dftest, la primera es para entrenar
# el modelo y la segunda para validarlo (validacion externa)

#%%

print(np.mean(y_test == y_pred_test))

#%%

# entrenamiento de variables numericas

clf = tree.DecisionTreeRegressor()

y = df['ArrDelay']
y_test = dftest['ArrDelay']

clf = clf.fit(x,y)
y_pred = clf.predict(x)
y_pred_test = clf.predict(x_test)

print('R cuadrado: ', r2_score(y, y_pred))
print('R cuadrado: ', r2_score(y_test, y_pred_test))


#%%

### RANDOM FOREST

# - Es un conjunto de cientos o de miles de arboles
# - Selecciona aleotoriamente observaciones de la base de datos
# - Selecciona aleatoriamente las variables

# Virtudes:
    # - evita sobreajustar los resultados del modelo
    # - asigna un peso variable a las distintas observaciones
    # - asigna un peso aleatorio a las distintas variables
    
# Resultados finales del modelo:
    # - usa la media de los resultados obtenidos en variables numéricas en los
    #   distintos arboles en caso de que sea la vraiable numérica.    
    # - o usa la categoría mayoritaria en caso de que la respuesta sea categorica
    
import pandas as pd
import numpy as np
import bz2
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo)

#%%

df = df.dropna(subset = ['ArrDelay'])
df = df.sample(frac = 1)
dftest = df.tail(500000)
df = df.head(500000)

#%%
# comparamos el metodo de arbol de clasificacion con el randomForest

#arbol de clasificacion
clf = tree.DecisionTreeClassifier()

x = df[['Distance','AirTime','DepTime','TaxiIn','TaxiOut','DepDelay']]
x_test = dftest[['Distance','AirTime','DepTime','TaxiIn','TaxiOut','DepDelay']]
y = df['ArrDelay'] > 0
y_test = dftest['ArrDelay'] > 0

clf = clf.fit(x,y)
y_pred_test = clf.predict(x_test)

#%%

print(np.mean(y_test == y_pred_test))

#%%
# RamdonForest
clf = RandomForestClassifier(n_estimators = 100, n_jobs = -1) #le pedimos que genere 100 arboles
clf.fit(x,y)
y_pred_test = clf.predict(x_test)

print(clf.feature_importances_) #clasificar la importancia de cada una de las
# variables que hemos usado, nos indica que variable es la más importante
# en la media de todos los arboles

#%%

print(np.mean(y_test == y_pred_test)) # incrementa el porcentaje de prediccion

#RandomForestRegressor() funcion para predecir variables numericas

#%%

### SUPORT VECTOR MACHINE

# Es un metodo de clasificacion probabilistica binaria, lineal o no lineal.
# Divide los grupos de puntos a través de una recta.
# En caso de que los datos no se puedan dividir en grupos, el método aplica
# una transformación del espacio de variables basado en kernel

import pandas as pd
import numpy as np
import bz2
from sklearn.svm import SVC

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo)

#%%

df = df.dropna(subset = ['ArrDelay'])
df = df.sample(frac = 1)
dftest = df.tail(1000)
df = df.head(1000)

#%%

x = df[['Distance','AirTime','DepTime','TaxiIn','TaxiOut','DepDelay']]
x_test = dftest[['Distance','AirTime','DepTime','TaxiIn','TaxiOut','DepDelay']]
y = df['ArrDelay'] > 10
y_test = dftest['ArrDelay'] > 10

clf = SVC(kernel = 'linear') # 'poly', 'rbf, 'sigmoid', 
clf.fit(x,y)
# si usamos kerner linear sería mejor llamar el modulo LinearSVC()
y_pred = clf.predict(x_test)

#%%

print(np.mean(y_test == y_pred))


#%%

# K-NEAREST NEIGHBOURS

# este método clasifica cada punto en una categoría basandose en la categoría
# de sus vecinos más cercanos

import pandas as pd
import numpy as np
import bz2
from sklearn.neighbors import KNeighborsClassifier

path = r'D:\DATOS\Users\jcgarciam\Downloads\archivos_base_python_data_science_big_data_esencial'

with bz2.open(path + '/base_datos_2008.csv.bz2', 'rt') as archivo:
    df = pd.read_csv(archivo, nrows = 1e6)

#%%

newdf = df[['AirTime','Distance','TaxiOut','ArrDelay']].dropna()
cols = newdf[newdf.columns[newdf.columns != 'ArrDelay']]

#%%

filtro = newdf['ArrDelay'] > 10

#%%

newdf['ArrDelay'][filtro] = 'Delayed'
newdf['ArrDelay'][filtro == False] = 'Not Delayed'

#%%

nbrs_3 = KNeighborsClassifier(n_neighbors = 3, n_jobs = -1) #miramos los 3
# vecinos mas cercanos
nbrs_3.fit(cols, newdf['ArrDelay'])

#%%

predicciones_3 = nbrs_3.predict(cols)

#%%

print(np.mean(predicciones_3 == newdf['ArrDelay']))

#%%

print(np.mean(newdf['ArrDelay'] == 'Not Delayed')) # el modelo incrementa como
# un 10% la prediccion al compararlo con la media

#%%

nbrs_1 = KNeighborsClassifier(n_neighbors = 1, n_jobs = -1)
#miramos el vecino mas cercano 

nbrs_1.fit(cols, newdf['ArrDelay'])
predicciones_1 = nbrs_1.predict(cols)
print(np.mean(predicciones_1 == newdf['ArrDelay']))
# mejora la predicción

#%%

print(np.mean(newdf['ArrDelay'] == 'Not Delayed'))

#%%

# revisamos la matrix de confusión

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(newdf['ArrDelay'], predicciones_1)
print(confusion_matrix) # por filas vemos los valores reales que toma la 
#variable, es decir los que se retrasaban y No se retrasaban
# y por columnas vemos la predicción de los que se retrasan y no se retrasan


#%%

### PYSPARK

from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster('local').setAppName('Mi programa')
sc = SparkContext(conf = conf)

#%%

print(sc)

#%%
lines = sc.textFile('ejemplo.txt')


#%%

### DATAFRAMES EN PYSPARK

from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster('local').setAppName('Mi programa')
sc = SparkContext(conf = conf)



from pyspark.sql.types import StringType
from pyspark import SQLContext
sqlContext = SQLContext(sc)

dfspark = sqlContext.read.format('csv').option('header', 'true').option('inferSchema', 'true').load(r'D:\DATOS\Users\jcgarciam\Downloads/base_datos_2008.csv')

#%%
print(dfspark.show(2))
print(dfspark.head(2))
print(dfspark.count())

#%%

dfspark = dfspark.sample(fraction = 0.001, withReplacement = False)
print(dfspark.count())

#%%

dfspark = dfspark.withColumn('ArrDelay', dfspark['ArrDelay'].cast('integer'))#transformamos la columna 'ArrDelay en formato entero

df2 = dfspark.na.drop(subset = ['ArrDelay','DepDelay','Distance']) #quitamos los vacíos de las columnas seleccionadas

#%%

df2 = df2.filter('ArrDelay is not NULL') #RECTIFICAMOS que las columnas no tengan NULLs

#%%

df2.printSchema() #funcion para saber el formato de las columnas

#%%

#podemos importar otras librerías externas a PySpark

import numpy as np

media = np.mean(df2.select('ArrDelay').collect()) #la función collect me devuelve toda la info que tenga la columna

#%%

df2.rdd.getNumPartitions() #función que nos muestra las particiones de nuestro ordenador con las que estamos trabajando

#%%

### TRANSFORMACIONES BÁSICAS EN PYSPARK

from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StringType
from pyspark import SQLContext
import numpy as np

conf = SparkConf().setMaster('local').setAppName('Mi programa')
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

dfspark = sqlContext.read.format('csv').option('header', 'true').option('inferSchema', 'true').load(r'D:\DATOS\Users\jcgarciam\Downloads/base_datos_2008.csv')
dfspark = dfspark.sample(fraction = 0.001, withReplacement = False)
dfspark = dfspark.withColumn('ArrDelay', dfspark['ArrDelay'].cast('integer'))
df2 = dfspark.na.drop(subset = ['ArrDelay','DepDelay','Distance'])
df2 = df2.filter('ArrDelay is not NULL')
df2 = df2.dropDuplicates()

#%%

df2.select('ArrDelay').filter('ArrDelay > 60').take(5) #seleccionamos los datos que cumplan la condicicion del filtro y nos muestre 5

df2.filter('ArrDelay > 60').take(5) #si queremos que nos muestre todos los registros que cumplan el filtro

#%%

media = np.mean(df2.select('ArrDelay').collect())
#le podemos aplicar una funcion a una columna usando el rdd que almacena la columna
df2.select('ArrDelay').rdd.map(lambda x: (x - media)**2).take(10)

#%%

#podemos usar groupbys

df2.groupBy('DayOfWeek').count().show()

#%%

df2.groupBy('DayOfWeek').mean('ArrDelay').show()

#%%
#podemos ver los elementos distintos de una columna
df2.select('Origin').rdd.distinct().take(5)

#%%
#podemos saber el número total de casos distintos que tenemos
df2.select('Origin').rdd.distinct().count()

#%%

### ACCIONES BÁSICAS EN PYSPARK

from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StringType
from pyspark import SQLContext
import numpy as np

conf = SparkConf().setMaster('local').setAppName('Mi programa')
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

dfspark = sqlContext.read.format('csv').option('header', 'true').option('inferSchema', 'true').load(r'D:\DATOS\Users\jcgarciam\Downloads/base_datos_2008.csv')
dfspark = dfspark.sample(fraction = 0.001, withReplacement = False)
dfspark = dfspark.withColumn('ArrDelay', dfspark['ArrDelay'].cast('integer'))
df2 = dfspark.na.drop(subset = ['ArrDelay','DepDelay','Distance'])
df2 = df2.filter('ArrDelay is not NULL')
df2 = df2.dropDuplicates()

#%%
# podemos generar un resumen estadístico
df2.select('ArrDelay').describe().show()

#%%
#podemos obtener una lista de ocurrencias para una variable categórica
df2.select('Origin').rdd.countsByValue() #aplicamos el método sobre un RDD no ditectamente sobre la columna


#%%
#podemos obtener el maximo sin usar el describe
df2.select('ArrDelay').rdd.max()

#%%
#podemos generar una lista
df2.select('Origin').rdd.collect()

#%%
# podemos generar una table de contigencias
df2.crosstab('DayOfWeek','Origin').take(2)


#%%

### OPERACIONES NUMÉRICAS CON RDD

from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StringType
from pyspark import SQLContext
import numpy as np

conf = SparkConf().setMaster('local').setAppName('Mi programa')
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

dfspark = sqlContext.read.format('csv').option('header', 'true').option('inferSchema', 'true').load(r'D:\DATOS\Users\jcgarciam\Downloads/base_datos_2008.csv')
dfspark = dfspark.sample(fraction = 0.001, withReplacement = False)
dfspark = dfspark.withColumn('ArrDelay', dfspark['ArrDelay'].cast('integer'))
df2 = dfspark.na.drop(subset = ['ArrDelay','DepDelay','Distance'])
df2 = df2.filter('ArrDelay is not NULL')
df2 = df2.dropDuplicates()

#%%
#comparación de la función sum con funciones lambda (el mismo resultado)
lista = sc.parallelize(range(1,1000000))
lista.reduce(lambda x, y: x + y)

lista.sum()

#%%

from pyspark.sql.functions import mean, stddev, col

media = df2.select(mean(col('ArrDelay'))).collect()
std = df2.select(stddev(col('ArrDelay'))).collect()

print('La media de los datos es: ', std[0][0])

#%%
# podemos crear una nueva columna por ejemplo, que se una operación entre las otras dos
df2.withColumn('Diferencia', df2['ArrDelay'] - df2['DepDelay']).collect()

#%%
#creamos una columna estandarizada
df2.withColumn('Standard', df2['ArrDelay'] - media[0][0]/std[0][0]).collect()

#%%

### JOIN EN PYSPARK

from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster('local').setAppName('Mi programa')
sc = SparkContext(conf = conf)

#%%

x = sc.parallelize([('a', 5), ('b', 6), ('c', 7), ('d', 8)])
y = sc.parallelize([('a', 1), ('a', 2), ('c', 3)])

#%%

x.join(y).collect() #hace un join inner
y.join(x).colect() #obtenemos el mismo resultado anterior
y.leftOuterJoin(x).collect() #le agregamos a 'y' lo que coincida con 'x'
x.leftOuterJoin(y).collect()
y.rightOuterJoin(x).collect()

#%%

### ACUMULADORES. CÓMO DETECTAR PATRONES EN NUESTROS DATOS

from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster('local').setAppName('Mi programa')
sc = SparkContext(conf = conf)
lines = sc.textFile('ejemplo.txt')

#lo acumuladores van incrementando o decrementan su valor. Podemos usarlos para detectar patrones

#%%

py = sc.accumulator(0) #función acumulador que empieza en cero
sp = sc.accumulator(0)

#%%
#creamnos una función que nos contabilice cuántas veces se habla de cada uno 'py' o 'sp'
def lenguajes(linea):
    global py, sp #son variables globales
    if 'Python' in linea:
        py += 1
        return True
    
        if 'Spark' in linea:
            sp += 1
            
    if 'Spark' in linea:
        sp += 1
        return True
    
    else:
        return False
    
valores = lines.filter(lenguajes)
#%%

valores.collect()

#%%

# CÓMO CONTRUIR FUNCIONES MAP

from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StringType
from pyspark import SQLContext

conf = SparkConf().setMaster('local').setAppName('Mi programa')
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

dfspark = sqlContext.read.format('csv').option('header', 'true').option('inferSchema', 'true').load(r'D:\DATOS\Users\jcgarciam\Downloads/base_datos_2008.csv')
dfspark = dfspark.sample(fraction = 0.001, withReplacement = False)
dfspark = dfspark.withColumn('ArrDelay', dfspark['ArrDelay'].cast('integer'))
df2 = dfspark.na.drop(subset = ['ArrDelay','DepDelay','Distance'])
df2 = df2.filter('ArrDelay is not NULL')
df2 = df2.dropDuplicates()

#%%
#las funciones map son aquellas que procesan nuestros datos almacenados en particiones
#aplicandoles filtros y ordenaciones y que posteriormente las funciones 'reduce' los procesen
# y resuman
A = sc.parallelize(df2.select('Origin').rdd.collect()) #creamos un objeto que es un array paralelizado

#%%

A.persist() #hemos guardado el objeto en memoria para que sea más rápido el cáculo

#%%
#la siguiente es una función de las más sencillas
mapfunction = A.map(lambda x: (x, 1)) #funcion que convierte el objeto en el mismo y un uno

#%%

mapfunction.collect()

#%%
#creamos una funcion similar a la anterior pero dando diferentes pesos
def fun(x):
    if x[0] in ['SEA','ATL','HOU']:
        return ((x, 2))
    elif x[0] == 'DEN':
        return ((x,3))
    else:
        return ((x, 1))

#%%

mapfunction2 = A.map(fun)

#%%

### CÓMO CONSTRUIR FUNCIONES REDUCE

from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StringType
from pyspark import SQLContext

conf = SparkConf().setMaster('local').setAppName('Mi programa')
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

dfspark = sqlContext.read.format('csv').option('header', 'true').option('inferSchema', 'true').load(r'D:\DATOS\Users\jcgarciam\Downloads/base_datos_2008.csv')
dfspark = dfspark.sample(fraction = 0.001, withReplacement = False)
dfspark = dfspark.withColumn('ArrDelay', dfspark['ArrDelay'].cast('integer'))

df2 = dfspark.na.drop(subset = ['ArrDelay','DepDelay','Distance'])
df2 = df2.filter('ArrDelay is not NULL')
df2 = df2.dropDuplicates()

#%%

A = sc.parallelize(df2.select('Origin').rdd.collect())

#%%

A.persist() #la convertimos en persistente para acelerar los procesos

#%%
#definimos una función map

mapfunction = A.map(lambda x: (x, 1))

#%%

def fun(x):
    if x[0] in ['SEA','ATL','HOU']:
        return((x, 2))
    elif x[0] == 'DEN':
        return ((x, 3))
    else:
        return ((x, 1))
    
#%%

mapfunction2 = A.map(fun)

#%%

#la función reduce más sencilla tiene la siguiente estructura
# 
reducefunction = mapfunction.reduceByKey(lambda x, y: x + y) #función para sumar los elementos

#%%
#para ver los resultados ejecutamos la siguiente línea:
    
reducefunction.collect() # ó reducefunction.take(10)

#%%
#usamos la otra función
reducefunction2 = mapfunction2.reduceByKey(lambda x, y: x + y)

#%%
#podemos ordenar los datos
reducefunction.sortByKey().take(10) #ordenamos por la llave alfabeticamente

#%%
#podemos ordenar por el campo sumado
reducefunction.sortBy(lambda x: x[1], ascending = False).take(10)

#%%

### EJEMPLOS BÁSICOS DE MAPREDUCE EN PYSPARK

from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StringType
from pyspark import SQLContext

conf = SparkConf().setMaster('local').setAppName('Mi programa')
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

#%%

lines = sc.textFile('ejemplo.txt') # cargamos el archivo de la página de Apache
lines.getNumPartitions()#nos indica la cantidad de particiones que necesitamos para ejecutar el proceso.      
# el objetivo es saber cuántas veces se habla de Spark o Python

#%%
# una aproximación para resolver el proble es usando acumuladores
py = sc.accumulator(0) #función acumulador que empieza en cero
sp = sc.accumulator(0)

def lenguajes(linea):
    global py, sp #son variables globales
    if 'Python' in linea:
        py += 1
        return True
    
        if 'Spark' in linea:
            sp += 1
            
    if 'Spark' in linea:
        sp += 1
        return True
    
    else:
        return False
    
valores = lines.filter(lenguajes)

#%%

valores.count() #nos dice en cuántas líneas ha ocurrido éste suceso
#si ejecutamos ésta línea más de una vez podemos tener datos erroneos en la
# variable py y sp
#%%
#podemos conbinar el ejemplo anterior con una función mapreduce para hacer un conteo
funcionmap = valores.map(lambda x: (x, 1))
contarvalores = funcionmap.reduce(lambda x, y: x + y)
contarvalores.count() #no cuenta filas repetidas

#%%

contarvalores.sortBy(lambda x: x[1], ascending = False).take(5) #podemos ver las repeticiones

#%%
# definimos una función que va a realizar lo mismo que los acumuladores, pero usando
# map y reduce
def lenguajes_map(x):
    if 'Python' in x and 'Spark' in x:
        return('Count', (1, 1))
    elif 'Python' in x:
        return('Count', (1, 0))
    elif 'Spark' in x:
        return('Count', (0, 1))
    else:
        return ('Count', (0, 0))
mapfun = lines.map(lenguajes_map)  

#%%

mapfun.count() #contabilizamos todo lo que hay

#%%
#podemos resumir la anterior función de la siguiente manera y no se va dañar
#si lo ejecutamos varias veces a comparación de los acumuladores
mapfun.reduceBy(lambda x, y: (x[0] + y[0], x[1])).collect()

#%%

from pyspark import __version__
print(__version__)
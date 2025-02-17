import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pickle
import numpy as np
import os
import json
import gzip

# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

def clean_data(data_df):
    df = data_df.copy()
    df = df.rename(columns={'default payment next month': 'default'})
    df = df.drop(columns='ID')
    df['EDUCATION'] = df['EDUCATION'].replace(0, np.nan)
    df['MARRIAGE'] = df['MARRIAGE'].replace(0, np.nan)
    df = df.dropna()
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
    return df

def get_features_target(data, target_column):
    x = data.drop(columns=[target_column])
    y = data[target_column]
    return x, y

def create_pipeline(df):
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    numerical_features = [col for col in df.columns if col not in categorical_features + ['default']]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )
    
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=21)),
            ('select_k_best', SelectKBest(f_classif, k=12)),
            ('model', SVC(C=0.8, kernel='rbf', gamma=0.1))
        ]
    )
    
    return pipeline

def optimize_hyperparameters(pipeline, x_train, y_train):
    param_grid = {
        'pca__n_components': [21],
        'select_k_best__k': [12],
        'model__C': [0.8],
        'model__kernel': ['rbf'],
        'model__gamma': [0.1],
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='balanced_accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)
    
    return grid_search.best_estimator_

def save_model(model):
    os.makedirs('files/models', exist_ok=True)
    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(model, f)

def calculate_metrics(model, x_train, y_train, x_test, y_test):
    metrics = []
    for dataset, x, y in [('train', x_train, y_train), ('test', x_test, y_test)]:
        y_pred = model.predict(x)
        metrics.append({
            'type': 'metrics',
            'dataset': dataset,
            'precision': round(precision_score(y, y_pred), 3),
            'balanced_accuracy': round(balanced_accuracy_score(y, y_pred), 3),
            'recall': round(recall_score(y, y_pred), 3),
            'f1_score': round(f1_score(y, y_pred), 3)
        })
    return metrics

def calculate_confusion_matrix(model, x_train, y_train, x_test, y_test):
    confusion_matrices = []
    for dataset, x, y in [('train', x_train, y_train), ('test', x_test, y_test)]:
        cm = confusion_matrix(y, model.predict(x))
        confusion_matrices.append({
            'type': 'cm_matrix',
            'dataset': dataset,
            'true_0': {'predicted_0': int(cm[0, 0]), 'predicted_1': int(cm[0, 1])},
            'true_1': {'predicted_0': int(cm[1, 0]), 'predicted_1': int(cm[1, 1])}
        })
    return confusion_matrices

def save_metrics(metrics, confusion_matrices):
    os.makedirs('files/output', exist_ok=True)
    with open('files/output/metrics.json', 'w') as f:
        json.dump(metrics + confusion_matrices, f, indent=4)

def main():
    train_data = pd.read_csv('files/input/train.csv')
    test_data = pd.read_csv('files/input/test.csv')
    
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)
    
    x_train, y_train = get_features_target(train_data, 'default')
    x_test, y_test = get_features_target(test_data, 'default')
    
    pipeline = create_pipeline(train_data)
    model = optimize_hyperparameters(pipeline, x_train, y_train)
    save_model(model)
    
    metrics = calculate_metrics(model, x_train, y_train, x_test, y_test)
    confusion_matrices = calculate_confusion_matrix(model, x_train, y_train, x_test, y_test)
    save_metrics(metrics, confusion_matrices)
    
if __name__ == '__main__':
    main()

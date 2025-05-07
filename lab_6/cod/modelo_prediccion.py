# Este codigo tiene como objetivo el generar modelos de prediccion
# en este caso calibrando y otros aspectos


# Importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Se procede a generar la clase

class ModeloPrediccion:
    def __init__(self, X, y):
        """
        Inicializa la clase ModeloPrediccion.

        :param df: DataFrame de pandas que contiene los datos.
        :param target: Nombre de la columna objetivo en el DataFrame.
        """
        self.X = X
        self.y = y
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, test_size=0.2, random_state=42):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.

        :param test_size: Proporción del conjunto de prueba.
        :param random_state: Semilla para la aleatoriedad.
        """
        
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)


    def train_model(self, model):
        """
        Entrena el modelo con los datos de entrenamiento.

        :param model: Modelo de sklearn a entrenar.
        """
        self.model = model
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate_model(self):
        """
        Evalúa el modelo utilizando el conjunto de prueba y devuelve las métricas.
        :return: Diccionario con las métricas del modelo.
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)  
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    

    def plot_voronoi(self):
        """
        Genera un gráfico de Voronoi para visualizar la clasificación del modelo.
        """
        from sklearn.decomposition import PCA
        from scipy.spatial import Voronoi, voronoi_plot_2d

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_train)

        vor = Voronoi(X_pca)
        fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=2, point_size=10)
        
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_train, s=50, cmap='viridis')
        plt.title('Voronoi Diagram')
        plt.show()

    def plot_roc_curve(self):
        """
        Genera la curva ROC para evaluar el rendimiento del modelo.
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        y_score = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

    def predict(self, X_new, probabilities=False):
        """
        Realiza predicciones sobre nuevos datos.

        :param X_new: Datos nuevos para predecir.
        :return: Predicciones del modelo.
        """
        if probabilities:
            return self.model.predict_proba(X_new)
        else:
            return self.model.predict(X_new)
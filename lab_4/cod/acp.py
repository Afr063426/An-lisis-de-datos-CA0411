

# Esta clase tiene como objetivo el ser un ejemplo de juguete para representar la programacion 
# orientada a objetos en Python. En este caso se tiene una representacion del analisis de 
# componentes principales (ACP) para un conjunto de datos.


# Se importa numpy ya que va ser necesario para hacer calculos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


    

    
class ACP_normado():

    #Se define el objeto y se indican aquellas cosas que el objeto necesita
    #para poder funcionar

    def __init__(self, data, D = None, M = None):
        self.__data = data
        if M is None:
            self.__M = np.diag(np.ones(data.shape[1]))
        else:
            self.__M = M
        if D is None:
            self.__D = 1/data.shape[0]*np.diag(np.ones(data.shape[0]))

        else:
            self.__D = D
        
        
        

    #Se define una propiedad que nos permite recuperar los datos 
    #que fueron pasados al objeto
    @property 
    def data(self):
        return self.__data
    

    #Se define una propiedad que nos permite recuperar los datos 
    #que fueron pasados al objeto
    @property 
    def M(self):
        return self.__M
        

    @property 
    def D(self):
        return self.__D
        


    

    # Se procede a hacer una funcion setter para los datos
    @data.setter
    def set_data(self, data):
        self.__data = data


    # Se procede a hacer una funcion setter para los datos
    @M.setter
    def set_M(self, M):
        self.__M = M


    @D.setter
    def set_D(self, D):
        self.__D = D


    # Se procede a definir algunas metricas
    # por ejemplo el promedio de las columnas
    # en este caso se puede hacer de esta forma 
    # dado que se estan usando un data frame de pandas
    # por lo que se pueden usar los metodos de pandas
    def mean(self):
        return self.__data.mean()
    
    # Se procede a definir la matriz de covarianza
    def cov(self):
        return self.__data.cov()
    

    # Se procede a estimar el ACP
    def pca(self):
        # En este caso primeramente se estandarizan los datos
        # para que tengan media 0 y varianza 1
        #self.__data_centered = (self.__data)#- self.__data.mean())
        
        # En este caso se estimara la matriz de correlaciones
        self.cov_matrix =  np.dot(self.__data.T, np.dot(self.__D, self.__data))

        #cov_matrix =  np.dot(np.dot(self.__data.T, self.__D), self.__data)

        #eig_vals, eig_vecs = np.linalg.eig(np.dot(self.cov_matrix , self.__M))
        # Se procede a estimar los autovalores y autovectores
        
        self.eig_vals, self.eig_vecs = np.linalg.eig(np.dot(self.cov_matrix, self.__M))

    
    # Se procede a estimar la varianza explicada
    # por las componentes principales
    def explained_variance(self, n_components):
        return self.eig_vals[:n_components] / sum(self.eig_vals)
    
    # Se procede a estimar las componentes principales
    def principal_components(self, n_components):
        self.components = np.dot(np.dot(self.__data, self.__M), self.eig_vecs[:, :n_components])
        return self.components
    

    #Se procede a realizar un grafico de las componentes principales
    # en este caso unicamente se van a escoger las componentes escogidas
    # por el usuario
    def plot(self, eje_x = 1, eje_y = 2):
        
        plt.scatter(self.components[:, eje_x-1], self.components[:, eje_y-1])
    
        for i, txt in enumerate(self.__data.index):
            plt.annotate(txt, (self.components[i, eje_x-1], self.components[i, eje_y-1]))
            
        plt.xlabel(f'Componente Principal {eje_x}')
        plt.ylabel(f'Componente Principal {eje_y}')
        plt.title('Componentes Principales')
        plt.show()

    #Se procede a realizar el circulo de correlaciones
    def circle_correlation(self, eje_x = 1, eje_y = 2):
        #Primero se debe estimar las correlaciones entre las variables y las componentes principales
        

        for i in self.__data.columns:
            if i == self.__data.columns[0]:
                self.correlation_circle = pd.DataFrame({"Variable": [i], 
                    f"Corr componente {eje_x}": [np.corrcoef(self.__data[i], self.components[:, eje_x-1])[0, 1]], 
                    f"Corr componente {eje_y}": [np.corrcoef(self.__data[i], self.components[:, eje_y-1])[0, 1]]})
            else:
                self.correlation_circle = pd.concat(
                    [self.correlation_circle,
                        pd.DataFrame({"Variable": [i], 
                        f"Corr componente {eje_x}": [np.corrcoef(self.__data[i], self.components[:, eje_x-1])[0, 1]], 
                        f"Corr componente {eje_y}": [np.corrcoef(self.__data[i], self.components[:, eje_y-1])[0, 1]]})
                    ]
                ) 
            
        #Se procede a graficar el circulo de correlaciones
        plt.scatter(self.correlation_circle[f"Corr componente {eje_x}"], self.correlation_circle[f"Corr componente {eje_y}"])
        for i, txt in enumerate(self.__data.columns):
            plt.annotate(txt, (self.correlation_circle.iloc[i, 1], self.correlation_circle.iloc[i, 2]))

        plt.xlabel(f'Correlacion con Componente Principal {eje_x}')
        plt.ylabel(f'Correlacion con Componente Principal {eje_y}')
        plt.title('Circulo de Correlaciones')
        plt.vlines(0, -1, 1)
        plt.hlines(0, -1, 1)
        plt.show()

        # #Se estima la calidad particular de los individuos
        # def quality_individuals(self, n_components):

        #     proyeccion_1 = 

    
        


class ACP_general(ACP_normado):
    
    
    #Se innicia el objeto en este caso ingresamos lo que se necesita en la nube de puntos
    def __init__(self, data, M):
        super().__init__(self, data)
        self.set_M(M)


class AFC(ACP_normado):
    
    #Se inicia el objeto en este caso vamos a requerir la tabla de contingencias y a partir de este vamos a generar las tablas de contingencias

    def __init__(self, data):
        M = np.linalg.inv(np.diag(data.T.sum(axis = 1)/data.to_numpy().sum()))    
        D = np.diag(data.sum(axis = 1)/data.to_numpy().sum())
        
        data_col = data.copy()
        data_col.loc["suma", :] = data_col.sum(axis = 0)
        data_col = data_col.drop(["suma"], axis = 0).div(data_col.loc["suma", :], axis = 1)
        
        data_fil = data.copy()
        data_fil.loc[:, "suma"] = data_fil.sum(axis = 1)
        data_fil = data_fil.drop(["suma"], axis = 1).div(data_fil["suma"], axis = 0)

        
        
        super().__init__(data_fil, D, M)
        
        

        self.data_ori = data
        self.data_col = data_col
        self.col = data_col.columns
        self.row = data_col.index
    
    

    def afc(self):
        self.pca()
        self.eig_vals, self.eig_vecs = self.eig_vals[1:], self.eig_vecs[:, 1:]

    def transicion(self): 
        self.components_col = np.empty((self.data_col.shape[1], self.components.shape[1]))
        
        for h in range(self.components.shape[1]):
            for j in range(self.data_col.shape[1]):
                self.components_col[j,h] =  (1/np.sqrt(self.eig_vals[h]))*np.sum(self.data_col.iloc[:, j]*self.components[:, h])

    def contribucion_pf(self, i, component):
        p_i = np.sum(self.data_ori.iloc[i-1,:])/np.sum(np.sum(self.data_ori))
        coord = self.components[i-1, component-1]**2
        return p_i*coord/self.eig_vals[component-1]
    
    def contribucion_col(self, j, component):
        q_j = np.sum(self.data_ori.iloc[:,j-1])/np.sum(np.sum(self.data_ori))
        coord = self.components_col[j-1, component-1]**2
        return q_j*coord/self.eig_vals[component-1]


    def plot_coords(self, axis_x = 1, axis_y = 2):
        #plt.axis([-1.25,1.25,-1.25,1.25])
        # plt.vlines(0, -1.25, 1.25)
        # plt.hlines(0, -1.25, 1.25)
        plt.scatter(self.components[:, axis_x-1], self.components[:, axis_y-1])
        for i, txt in enumerate(self.row):
            plt.annotate(txt, (self.components[i, axis_x-1], self.components[i, axis_y-1]))

        plt.scatter(self.components_col[:, axis_x-1], self.components_col[:, axis_y-1])
        for i, txt in enumerate(self.col):
            plt.annotate(txt, (self.components_col[i, axis_x-1], self.components_col[i, axis_y-1]))
        
        plt.title('Plano factorial')
        plt.show()

        

    



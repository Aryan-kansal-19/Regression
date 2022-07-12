import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import pickle


df = pd.read_csv("..\..\datasets\Linear\Iris\Iris.csv")
X = df.drop(["Id", "Species"], axis = 1)
y = df["Species"]

logreg = LogisticRegression()
#xg = XGBClassifier(random_state = 0)
knn = KNeighborsClassifier(n_neighbors = 8)

model_dict = {'Logistic Regression': 'Iris_Logistic.pkl', 'K Nearest Neighbour': 'Iris_Knn.pkl'}


st.title("IRIS Pridiction Model Using Different Algorithms")

st.subheader("Select Model to Apply on Dataset")

model = st.selectbox('Model', ('Logistic Regression', 'K Nearest Neighbour',));
Accuracy_dict = {'Logistic Regression': '97.3%', 'K Nearest Neighbour': '97.7%', 'XGBoost': '95%'}

st.subheader("Accuracy Score of "+model+" Model is "+Accuracy_dict[model])

try: 
    pickle_model = pickle.load(open(model_dict[model], 'rb'))
except:
    logreg.fit(X, y)
    knn.fit(X, y)
    #xg.fit(X, y)

    pickle.dump(logreg, open('Iris_Logistic.pkl', 'wb'))
    #pickle.dump(xg, open('Iris_XGB.pkl', 'wb'))
    pickle.dump(knn, open('Iris_Knn.pkl', 'wb'))
     
    pickle_model = pickle.load(open(model_dict[model], 'rb'))


st.subheader("Please Enter Values")
sepallg = st.slider("Sepal Length", 0.0, 8.0, 5.0)

sepalwd = st.slider("Sepal Width", 0.0, 4.5, 2.3)

petallg = st.slider("Petal Length", 0.0, 7.0, 4.0)

petalwd = st.slider("Petal Width", 0.0, 2.5, 1.3)


if(st.button("Predict Species")):
    st.subheader(pickle_model.predict([[sepallg, sepalwd, petallg, petalwd]]))
    
st.subheader("Downloadables")

col1, col2, col3= st.columns([5, 4, 2])

col1.write("Download Jupyter Notebook of this Project")
with open("Iris_Classification.ipynb", 'rb') as file:
    if(col2.download_button("Download", data = file, file_name = "Iris_Classification.ipynb", mime = "JupyterNotebook/ipynb")):
        col3.write("Jupyter Notebook Downloaded")

col1.write("Download Dataset")
with open("..\..\datasets\Linear\Iris\Iris.csv", "rb") as file:
    if(col2.download_button("Download", data = file, file_name = "Iris.csv", mime = "text/csv")):
        col3.write("Dataset Downloaded")

col1.write("Download Logistic Regression Model")
with open("Iris_Logistic.pkl", "rb") as file:
    if(col2.download_button("Download", data = file, file_name = "Iris_logistic.pkl", mime = "text/pkl")):
        col3.write("Pickle Model Downloaded")
        
col1.write("Download KNN model")
with open("Iris_Knn.pkl", "rb") as file:
    if(col2.download_button("Download", data = file, file_name = "Iris_knn.pkl", mime = "text/pkl")):
        col3.write("Pickle Model Downloaded")







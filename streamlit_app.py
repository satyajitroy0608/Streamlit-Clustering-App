# importing libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as kmeans
from sklearn.decomposition import PCA # principal component analysis
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

# display a heading
st.header("Report on clustering")

# Go to anaconda prompt to run this app
# cd PycharmProjects/DataScienceApp
# streamlit run streamlit_app.py

# reading the dataset
data = pd.read_csv(r"C:\Users\Admin\Desktop\Clustering_Dataset.csv",index_col='Unnamed: 0',nrows=1000)
# nrows =1000 so that it reads only first 1000 rows at a time, as it is a very large dataset
# preprocessing the dataset
data=pd.get_dummies(data,columns=['Gender','Customer Type','Type of Travel','Class','satisfaction'])
data.drop(['id','Gender_Female','Customer Type_disloyal Customer','Type of Travel_Business travel','satisfaction_neutral or dissatisfied'],axis=1,inplace=True)
data.drop(['Arrival Delay in Minutes'],axis=1,inplace=True)

# displaying the dataframe
st.subheader("The processed dataframe")
st.dataframe(data)

#scaling the data
scaler = StandardScaler()
scaled_data=scaler.fit_transform(data)

# creating and fitting the model
model = kmeans(n_clusters=2,n_init=25).fit(scaled_data)

# initialising the PCA
pca=PCA(n_components=2)

# fitting the PCA
principalComponents=pca.fit_transform(scaled_data)

# making a dataframe of the principal components
principalDf=pd.DataFrame(data=principalComponents,columns=['principal_component_1','principal_component_2'])

# displaying the PCA dataframe
st.subheader("The dataframe showing the 2 principle components")
st.dataframe(principalDf)

# visualisation of the clusters
fig_cluster = px.scatter(principalDf,x='principal_component_1',y='principal_component_2',color=model.labels_)

# displaying the clusters
st.subheader("Displaying the clusters")
st.plotly_chart(fig_cluster)

# list to store the within sum of squared error for the different clusters given the respective cluster size
wss = []

# loop to iterate over the no. of clusters and calculate the wss
for i in range(1,11):
    # kmeans
    fitx = kmeans(n_clusters=i, init='random', n_init=5, random_state=109).fit(scaled_data)
    # appending the value
    wss.append(fitx.inertia_)

# making the matplotlib figure
fig,ax = plt.subplots(figsize=(11,8.5))
plt.plot(range(1,11),wss,'bx-')
plt.xlabel('Number of clusters $k$')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal $k$')

# displaying the elbow curve
st.subheader("The Elbow Curve")
st.pyplot(fig)

# user interactivity
st.subheader("Getting the input from the user")
input = st.text_area("Input your values")

try:
    cleaned_input = [float(i.strip()) for i in input.split(", ")]
    output = model.predict(np.array(cleaned_input).reshape(1,24))

    st.text("The closest cluster to the data is {}".format(output[0]))


except:
    pass

# st.balloons()


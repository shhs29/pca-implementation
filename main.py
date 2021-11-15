import numpy
import pandas as pd
import numpy as np
import plotly.express as px
import copy
from sklearn.metrics import mean_squared_error
from scipy import linalg

with open('../iris.data') as file:
    data = file.readlines()

cleaned_data = list()
for data_item in data:
    if len(data_item) >= 2:
        items = data_item[:-1].split(',')
        items = [float(x) for x in items[:-1]] + [items[-1]]
        cleaned_data.append(items)

data = pd.DataFrame(cleaned_data, columns=['feat_1', 'feat_2', 'feat_3', 'feat_4', 'label'])

values = data.to_numpy()
attributes = copy.deepcopy(values)
features = 4
mean = list()

for i in range(features):
    mean.insert(i, np.mean(values[:, i]))
    attributes[:, i] = values[:, i] - mean[i]

labels = attributes[:, 4]
x = attributes[:, 0:4]
covariance_matrix = numpy.cov(x.astype(float), rowvar=False)
k1 = 2
k2 = 3
eig_values1, eig_vectors1 = linalg.eigh(covariance_matrix, eigvals=(4 - k1, 4 - 1))
eig_values2, eig_vectors2 = linalg.eigh(covariance_matrix, eigvals=(4 - k2, 4 - 1))

projection_matrix1 = eig_vectors1
projection_matrix2 = eig_vectors2

projection_data1 = numpy.dot(x, projection_matrix1)
projection_data2 = numpy.dot(x, projection_matrix2)

# Plotting PCA for K=2
fig1 = px.scatter(projection_data1, x=0, y=1, color=values[:, 4], labels={'0': 'PC 1', '1': 'PC 2'})
fig1.show()

# Plotting PCA for K=3
fig2 = px.scatter_3d(
    projection_data2, x=0, y=1, z=2, color=values[:, 4],
    title=f'PCA',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
fig2.show()

# error calculation for different values of K
for k in range(4):
    eig_values, eig_vectors = linalg.eigh(covariance_matrix, eigvals=(4 - (k + 1), 4 - 1))

    projection_matrix = eig_vectors

    projection_data = numpy.dot(x, projection_matrix)

    pred = projection_data.dot(eig_vectors.T) + mean
    error = mean_squared_error(x, pred)
    print(error)

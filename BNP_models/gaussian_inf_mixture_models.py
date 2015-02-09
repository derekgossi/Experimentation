import numpy as np
import random
import pandas as pd
from sklearn.mixture import DPGMM
import matplotlib.pyplot as plt
import seaborn as sb

# Generate some toy multivariate gaussians
gaussian_1 = np.random.multivariate_normal([0, 0],[[5, 0], [0, 1]], 200)
gaussian_2 = np.random.multivariate_normal([5,5],[[1, 0], [0, 1]], 100)
gaussian_3 = np.random.multivariate_normal([-4, -4],[[1, 3], [0, 1]], 124)
gaussian_4 = np.random.multivariate_normal([-2, 7],[[1, 0], [0, 1]], 59)
gaussians = [gaussian_1, gaussian_2, gaussian_3, gaussian_4]

# Put them in a dataframe
for num, gaussian in enumerate(gaussians):
    df = pd.DataFrame({'x' : gaussian[:,0], 'y' : gaussian[:,1]})
    if num == 0: gaussian_data = df
    else: gaussian_data = pd.concat([gaussian_data, df], ignore_index=True)

# Plot the toy data
colors = ['r', 'b', 'y', 'g']

for gaussian, c in zip(gaussians, colors):
    plt.scatter(gaussian[:,0], gaussian[:,1], color=c)

plt.xlim([-12,12])
plt.ylim([-12,12])
plt.show()

# Shuffle dataframe
gaussian_data = gaussian_data.reindex(np.random.permutation(gaussian_data.index))

# Choose a max number of components for the algorithm
max_components = 8

# Count the number of clusters the DPGMM chooses
num_clusters = []
size_sample = []

# Try clustering at different sample sizes
for iteration in range(int(np.floor(len(gaussian_data) / 10)) - 2):
    # Number of samples to use
    max_sample_value = ((iteration + 2) * 10) 
    sample_set = gaussian_data[0:max_sample_value]
    size_sample.append(max_sample_value - 0)
    
    # Fit Dirichlet Process Gaussian Mixture Model
    dpgmm_model = DPGMM(n_components = max_components, n_iter=1000, alpha=1.0)
    fitted_dpgmm = dpgmm_model.fit(sample_set)
    dpgmm_predictions = fitted_dpgmm.predict(gaussian_data)
    num_clusters.append(len(set(dpgmm_predictions)))
    
    # Append predicted labels to dataframe
    gaussian_data['predicted'] = dpgmm_predictions

# Give a unique color to each category
unique_categories = list(set(gaussian_data['predicted']))
color_labels = ['r', 'y', 'g', 'b', 'c', 'm', 'k', 'w']
colors = [color_labels[unique_categories.index(i)] for i in gaussian_data['predicted']]

# Plot predicted data
plt.scatter(gaussian_data['x'], gaussian_data['y'], c=colors)
plt.xlim([-12,12])
plt.ylim([-12,12])
plt.show()

# Plot how many clusters DPGMM chose based on size of dataset
plt.plot(size_sample, num_clusters)
plt.xlabel('Sample Size')
plt.ylabel('Number of Clusters')
plt.title('Dirichlet Process Gaussian Mixture Model Predicted Clusters by Sample Size')
plt.show()


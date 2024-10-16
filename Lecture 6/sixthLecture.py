# K-Means Clustering
# Exam Material ---

import pandas as pd
from fontTools.agl import AGL2UV
from pygments.styles.paraiso_dark import ORANGE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# კლასტერების რაოდენობას ვსაზღვრავთ ვიზუალიდან
model = KMeans(n_clusters=5)
data = pd.read_csv('Mall Customers.csv')
data.drop(['CustomerID', 'Gender', 'Age'], axis=1, inplace=True)

data['Annual Income (k$)'] = StandardScaler().fit_transform(data[['Annual Income (k$)']])
data['Spending Score (1-100)'] = StandardScaler().fit_transform(data[['Spending Score (1-100)']])

# K-Mean

print(data.head())

# Clustering
model.fit(data.values)
labels = model.predict(data.values)

print(model.inertia_) # Output : 65.56840815571681 [თუ n_clusters=5]

plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=labels)
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=80,c='black')
plt.show()

# Dendograma
Z = linkage(data.values, method='ward')
dendrogram(Z)
plt.show()

# AgglomerativeClustering
model1 = AgglomerativeClustering(n_clusters=5)
labels1 =model1.fit_predict(data.values)
print(labels1)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data = pd.read_csv('Iris no labels.csv')

# 4 განზომილებიდან გადაყავს 2 გაზომილებაში
tranformer = TSNE(n_components=2, perplexity=30)
data = tranformer.fit_transform(data.values)

# data[:, 0], data[:, 1] -> ნიშნავს ყველა სვეტს და ყველა რიგს
plt.scatter(data[:, 0], data[:, 1])

plt.show()
print(data.shape)

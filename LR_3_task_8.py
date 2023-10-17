import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target
# створення об'єкту K-Means з вказаними параметрами для подальшої кластеризації даних
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
                verbose=0, random_state=None, copy_x=True, algorithm='auto')
#обчислення к-середнього кластеризуванння
kmeans.fit(X)
# Обчислюэкластерні центри та передбаэ індекс кластера для кожного зразка.
y_kmeans = kmeans.predict(X)

plt.figure()
# візуалізація результатів кластерізації з пошуком 5 кластерів
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

# метод приймає набір даних - Х, кількість кластерів, що шукаємо - n_clusters та rseed для генерації випадкових чисел
def find_clusters(X, n_clusters, rseed=2):
    # створення {n_clusters} випадкових центрів кластерів з точок набору даних
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # оцінка приналежності точки до кожного центру
        labels = pairwise_distances_argmin(X, centers)
        # обчислення нового центру кластера як середньогог значення всіх точок кластеру
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        # якщо усі нові центри та старі ідентичні - цикл завершується 
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels


centers, labels = find_clusters(X, 3)
#візуалізація результатів кластерізації методом find_clusters() з трьома кластерами і значенням для генератора випадкових чисел 2
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
centers, labels = find_clusters(X, 3, rseed=0)
#візуалізація результатів кластерізації методом find_clusters() з трьома кластерами і значенням для генератора випадкових чисел 0
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
#візуалізація результатів кластерізації з трьома кластерами 
labels = KMeans(3, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

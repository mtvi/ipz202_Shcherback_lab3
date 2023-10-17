import json
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn import covariance, cluster

# Вхідний файл із символічними позначеннями компаній
input_file = 'company_symbol_mapping.json'

# Завантаження прив'язок символів компаній до їх повних назв
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

# Завантаження архівних даних котирувань
start_date = "2003-07-03"
end_date = "2007-05-05"
quotes = [yf.download(symbol, start=start_date, end=end_date) for symbol in symbols]

# Вилучення котирувань, що відповідають відкриттю та закриттю біржі
opening_quotes = (np.array([quote.Open for quote in quotes if len(quote.Open) > 0]).astype(float))

closing_quotes = (np.array([quote.Close for quote in quotes if len(quote.Close) > 0]).astype(float))

# Обчислення різниці між двома видами котирувань
quotes_diff = closing_quotes - opening_quotes

X = quotes_diff.copy().T
X /= X.std(axis=0)

# Створення моделі графа
edge_model = covariance.GraphicalLassoCV()

# Навчання моделі
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Створення моделі кластеризації на основі поширення подібності
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

for i in range(num_labels + 1):
    print('Cluster', i + 1, '==>', ', '.join([names[j] for j, label in enumerate(labels) if label == i]))
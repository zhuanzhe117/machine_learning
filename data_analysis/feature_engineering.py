from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
# print iris.data
# print iris.target

newData = StandardScaler().fit_transform(iris.data)

from sklearn.preprocessing import MinMaxScaler
MinMaxScaler().fit_transform(iris.data)

from sklearn.preprocessing import Normalizer
print Normalizer().fit_transform(iris.data)
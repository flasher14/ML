from pandas import read_csv
from numpy import unique

path = './oil-spill.csv'
df = read_csv(path, header=None)

print(df.nunique())

------------------------------------------

from numpy import loadtxt
from numpy import unique
path = './oil-spill.csv'

data = loadtxt(path, delimiter=',')

for i in range(data.shape[1]):
 num = len(unique(data[:, i]))
 percentage = float(num) / data.shape[0] * 100
 print('%d, %d, %.1f%%' % (i, num, percentage))

--------------------------------------------

# summarize the percentage of unique values for each column using numpy
from urllib.request import urlopen
from numpy import loadtxt
from numpy import unique

path = './oil-spill.csv'
data = loadtxt(path, delimiter=',')

for i in range(data.shape[1]):
    num = len(unique(data[:, i]))
    percentage = float(num) / data.shape[0] * 100
    if percentage < 1:
        print('%d, %d, %.1f%%' % (i, num, percentage))

------------------------------------------------

# delete columns where number of unique values is less than 1% of the rows
from pandas import read_csv
path = './oil-spill.csv'

df = read_csv(path, header=None)
print(df.shape)
counts = df.nunique()

to_del = [i for i,v in enumerate(counts) if (float(v)/df.shape[0]*100) < 1]
print(to_del)
df.drop(to_del, axis=1, inplace=True)
print(df.shape)

----------------------------------------------------

from numpy import arange
from pandas import read_csv
from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot
path = './oil-spill.csv'
df = read_csv(path, header=None)
data = df.values
X = data[:, :-1]
y = data[:, -1]
print(X.shape, y.shape)
thresholds = arange(0.0, 0.55, 0.05)
results = list()
for t in thresholds:
    transform = VarianceThreshold(threshold=t)
    X_sel = transform.fit_transform(X)
    n_features = X_sel.shape[1]
    print('>Threshold=%.2f, Features=%d' % (t, n_features))
    results.append(n_features)
pyplot.plot(thresholds, results)
pyplot.show()

--------------------------------------------------------

#list all duplicates
from pandas import read_csv
path = './oil-spill.csv'
df = read_csv(path, header=None)
dups = df.duplicated()
print(dups.any())
print(df[dups])

-----------------------------------------------------

# delete rows of duplicate data from the dataset
from pandas import read_csv
path = './oil-spill.csv'
df = read_csv(path, header=None)
print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)
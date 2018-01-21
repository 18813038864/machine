import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
# X = np.arange(-5.0, 5.0, 0.1)
# Y = np.arange(-5.0, 5.0, 0.1)
#
# x, y = np.meshgrid(X, Y)
# f = 17 * x ** 2 - 16 * np.abs(x) * y + 17 * y ** 2 - 225
#
# fig = plt.figure()
# cs = plt.contour(x, y, f, 0, colors='r')
# plt.show()
# sw = stopwords.words("english")
# print sw[0]
# print len(sw)

stemmer = SnowballStemmer("english")

print stemmer.stem("responsiveness")
print stemmer.stem("responsivity")
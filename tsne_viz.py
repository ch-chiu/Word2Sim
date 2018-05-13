from sklearn.externals import joblib
import os
import json
from time import time
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from gensim.models import Doc2Vec

import colorsys
N = 12
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
color = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
color = list(color)
#color = ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628")

t0 = time()
model = Doc2Vec.load('model/imdb_doc2vec.m')
X = model.docvecs.doctag_syn0
y = model.docvecs.doctags
y = np.array([(label[0]) for label in y])

# model = joblib.load('model/20newsgroup_average.m')
# X = model[0]
# y = np.array(model[1])

print("training t-sne model...")
X = PCA(n_components=50).fit_transform(X)
X = manifold.TSNE(n_components=2, verbose=2).fit_transform(X)

print("use {} seconds".format(time() - t0))

doc_list = []
out_file = "data/tsne.json"
for i in range(0, len(y)):
    doc_0 = dict()
    doc_0['x'] = str(X[i][0])
    doc_0['y'] = str(X[i][1])
    doc_0['label'] = str(y[i])
    doc_list.append(doc_0)

with open(out_file, 'w') as f:
    json.dump(doc_list, f, indent=3)
f.close()

# with open('model/tsne.json') as f:
#     data = json.load(f)
# x = [d['x'] for d in data]
# y = [d['y'] for d in data]
# label = [d['label'] for d in data]

plt.figure()
# plt.scatter(x, y, c=label, cmap='jet')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet')
plt.colorbar()
plt.show()
plt.title("t-SNE visualization of ESX bug data")
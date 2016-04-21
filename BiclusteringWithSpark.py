import re, operator
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import dia_matrix
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import assert_all_finite, check_array
from sklearn.utils.arpack import eigsh, svds
from sklearn.externals.six import iteritems

#Loading and Vectorizing the Data

def number_aware_tokenizer(doc):
    token_pattern = re.compile(u'(?u)\\b\\w\\w+\\b')
    tokens = token_pattern.findall(doc)
    tokens = ["#NUMBER" if token[0] in "0123456789_" else token
              for token in tokens]
    return tokens

print("Loading the 20 newsgroup Dataset")
categories = ['alt.atheism', 'comp.graphics',
              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'comp.windows.x', 'misc.forsale', 'rec.autos',
              'rec.motorcycles', 'rec.sport.baseball',
              'rec.sport.hockey', 'sci.crypt', 'sci.electronics',
              'sci.med', 'sci.space', 'soc.religion.christian',
              'talk.politics.guns', 'talk.politics.mideast',
              'talk.politics.misc', 'talk.religion.misc']
newsgroups = fetch_20newsgroups(categories=categories)
print("Vectorizing...")
vectorizer = TfidfVectorizer(stop_words='english', min_df=5,tokenizer=number_aware_tokenizer)
X = vectorizer.fit_transform(newsgroups.data)

#Spectral Coclustering Algorithm

def custom_svd(array, n_components, n_discard,n_svd_vecs):
	u, _, vt = svds(array, k=n_components, ncv=n_svd_vecs)
	if np.any(np.isnan(vt)):
		_, v = eigsh(safe_sparse_dot(array.T, array),ncv=n_svd_vecs)
		vt = v.T
	if np.any(np.isnan(u)):
		_, u = eigsh(safe_sparse_dot(array, array.T),ncv=n_svd_vecs)
	assert_all_finite(u)
	assert_all_finite(vt)
	u = u[:, n_discard:]
	vt = vt[n_discard:]
	return u, vt.T

def scale_normalize(X):
	row_diag = np.asarray(1.0 / np.sqrt(X.sum(axis=1))).squeeze()
	col_diag = np.asarray(1.0 / np.sqrt(X.sum(axis=0))).squeeze()
	row_diag = np.where(np.isnan(row_diag), 0, row_diag)
	col_diag = np.where(np.isnan(col_diag), 0, col_diag)
	if issparse(X):
		n_rows, n_cols = X.shape
		r = dia_matrix((row_diag, [0]), shape=(n_rows, n_rows))
		c = dia_matrix((col_diag, [0]), shape=(n_cols, n_cols))
		an = r * X * c
	else:
		an = row_diag[:, np.newaxis] * X * col_diag
	return an, row_diag, col_diag

def SpectralCoclustering(X,n_clusters, n_jobs, random_state,n_svd_vecs):
	normalized_data, row_diag, col_diag = scale_normalize(X)
	n_sv = 1 + int(np.ceil(np.log2(n_clusters)))
	u, v = custom_svd(normalized_data, n_sv,1,n_svd_vecs)
	z = np.vstack((row_diag[:, np.newaxis] * u,col_diag[:, np.newaxis] * v))
	model = KMeans(n_clusters, n_jobs=n_jobs,random_state=random_state)
	model.fit(z)
	labels = model.labels_
	n_rows = X.shape[0]
	row_labels = labels[:n_rows]
	column_labels = labels[n_rows:]
	rows = np.vstack(row_labels == c for c in range(n_clusters))
	columns = np.vstack(column_labels == c for c in range(n_clusters))
	return rows,columns,row_labels

print("CoClustering")
cocluster= SpectralCoclustering(X,n_clusters=len(categories),n_jobs=1,random_state=0,n_svd_vecs=None)

#Printing the data
feature_names = vectorizer.get_feature_names()
document_names = list(newsgroups.target_names[i] for i in newsgroups.target)
def bicluster_ncut(i):
	rows= np.nonzero(cocluster[0][i])[0]
	cols= np.nonzero(cocluster[1][i])[0]
	if not (np.any(rows) and np.any(cols)):
		import sys
		return sys.float_info.max
	row_complement = np.nonzero(np.logical_not(cocluster[0][i]))[0]
	col_complement = np.nonzero(np.logical_not(cocluster[1][i]))[0]
	weight = X[rows][:, cols].sum()
	cut = (X[row_complement][:, cols].sum() +X[rows][:, col_complement].sum())
	return cut / weight

def most_common(d):
    return sorted(iteritems(d), key=operator.itemgetter(1), reverse=True)

bicluster_ncuts = list(bicluster_ncut(i) for i in range(len(newsgroups.target_names)))
best_idx = np.argsort(bicluster_ncuts)[:5]

print("Best biclusters:")

for idx, cluster in enumerate(best_idx):
	n_rows = X.shape[0]
	n_cols = X.shape[1]
	cluster_docs, cluster_words = np.nonzero(cocluster[0][cluster])[0],np.nonzero(cocluster[1][cluster])[0]
	if not len(cluster_docs) or not len(cluster_words):
		continue
	counter = defaultdict(int)
	for i in cluster_docs:
		counter[document_names[i]] += 1
	cat_string = ", ".join("{:.0f}% {}".format(float(c) / n_rows * 100, name)for name, c in most_common(counter)[:3])
	out_of_cluster_docs = cocluster[2] != cluster
	out_of_cluster_docs = np.where(out_of_cluster_docs)[0]
	word_col = X[:, cluster_words]
	word_scores = np.array(word_col[cluster_docs, :].sum(axis=0)-word_col[out_of_cluster_docs, :].sum(axis=0))
	word_scores = word_scores.ravel()
	important_words = list(feature_names[cluster_words[i]]for i in word_scores.argsort()[:-11:-1])
	print("bicluster {} : {} documents, {} words".format(idx, n_rows, n_cols))
	print("categories   : {}".format(cat_string))
	print("words        : {}\n".format(', '.join(important_words)))

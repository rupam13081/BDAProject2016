from collections import defaultdict
import operator
import re
from time import time
import numpy as np
from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from splearn.cluster import KMeans
from scipy.sparse import dia_matrix
from sklearn.externals.six import iteritems
from scipy.sparse import issparse
from scipy.linalg import eigh as largest_eigh
#from sklearn.utils.arpack import eigsh, svds
from sklearn.utils.validation import assert_all_finite, check_array
from splearn.decomposition import SparkTruncatedSVD
from splearn.decomposition.truncated_svd import svd, svd_em

def number_aware_tokenizer(doc):
    token_pattern = re.compile(u'(?u)\\b\\w\\w+\\b')
    tokens = token_pattern.findall(doc)
    tokens = ["#NUMBER" if token[0] in "0123456789_" else token
              for token in tokens]
    return tokens

print("Fetching Data...")
categories = ['alt.atheism','comp.graphics']
newsgroups = fetch_20newsgroups(categories=categories)

from splearn.rdd import ArrayRDD
from splearn.feature_extraction.text import SparkHashingVectorizer
from splearn.feature_extraction.text import SparkTfidfTransformer
from splearn.pipeline import SparkPipeline

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

X_rdd = ArrayRDD(sc.parallelize(newsgroups.data, 4))  # sc is SparkContext

dist_pipeline = SparkPipeline((
    ('vect', SparkHashingVectorizer()),
    ('tfidf', SparkTfidfTransformer( min_df=5,
                             tokenizer=number_aware_tokenizer))
))

result_dist = dist_pipeline.fit_transform(X_rdd)  # SparseRDD

from splearn.cluster import SparkKMeans

dist = SparkKMeans(n_clusters=4, init='k-means++', random_state=42)
dist.fit(X_rdd)



n_svd_vecs=None



def svd(blocked_rdd, k):
    """
    Calculate the SVD of a blocked RDD directly, returning only the leading k
    singular vectors. Assumes n rows and d columns, efficient when n >> d
    Must be able to fit d^2 within the memory of a single machine.
    Parameters
    ----------
    blocked_rdd : RDD
        RDD with data points in numpy array blocks
    k : Int
        Number of singular vectors to return
    Returns
    ----------
    u : RDD of blocks
        Left eigenvectors
    s : numpy array
        Singular values
    v : numpy array
        Right eigenvectors
    """

    # compute the covariance matrix (without mean subtraction)
    # TODO use one func for this (with mean subtraction as an option?)
    c = blocked_rdd.map(lambda x: (x.T.dot(x), x.shape[0]))
    prod, n = c.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))

    # do local eigendecomposition
    w, v = ln.eig(prod / n)
    w = np.real(w)
    v = np.real(v)
    inds = np.argsort(w)[::-1]
    s = np.sqrt(w[inds[0:k]]) * np.sqrt(n)
    v = v[:, inds[0:k]].T

    # project back into data, normalize by singular values
    u = blocked_rdd.map(lambda x: np.inner(x, v) / s)

    return u, s, v



def _svd(array, n_components, n_discard):
	print("alsjdhakldjf haifshc alisfh ciadsufh lsdjfh dlksjafhsdlkfjh")
	c = array.map(lambda x: (x.T.dot(x), x.shape[0]))
	prod, n = c.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))

	# do local eigendecomposition
	w, v = ln.eig(prod / n)
	w = np.real(w)
	v = np.real(v)
	inds = np.argsort(w)[::-1]
	s = np.sqrt(w[inds[0:k]]) * np.sqrt(n)
	v = v[:, inds[0:k]].T

	# project back into data, normalize by singular values
	u = blocked_rdd.map(lambda x: np.inner(x, v) / s)
	u, _, vt = svd(array, 1)
	if np.any(np.isnan(vt)):
		# some eigenvalues of A * A.T are negative, causing
		# sqrt() to be np.nan. This causes some vectors in vt
		# to be np.nan.
		_, v = eigsh(safe_sparse_dot(array.T, array),
				ncv=n_svd_vecs)
		vt = v.T
	if np.any(np.isnan(u)):
		_, u = eigsh(safe_sparse_dot(array, array.T),
				ncv=n_svd_vecs)
	assert_all_finite(u)
	assert_all_finite(vt)
	u = u[:, n_discard:]
	vt = vt[n_discard:]
	return u, vt.T

if issparse(X):
		n_rows, n_cols = X.shape
		r = dia_matrix((row_diag, [0]), shape=(n_rows, n_rows))
		c = dia_matrix((col_diag, [0]), shape=(n_cols, n_cols))
		an = r * X * c
else:
	an = row_diag[:, np.newaxis] * X * col_diag
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
normalized_data, row_diag, col_diag = scale_normalize(X)
n_sv = 1 + int(np.ceil(np.log2(n_clusters)))
	
def SpectralCoclustering(X,n_clusters, svd_method,n_svd_vecs, mini_batch, init, n_init, n_jobs, random_state):
	normalized_data, row_diag, col_diag = scale_normalize(X)
	n_sv = 1 + int(np.ceil(np.log2(n_clusters)))
	u, v = _svd(normalized_data, n_sv, n_discard=1)
	z = np.vstack((row_diag[:, np.newaxis] * u,col_diag[:, np.newaxis] * v))
	if mini_batch:
		model = MiniBatchKMeans(n_clusters,random_state=random_state)
	else:
		model = SparkKMeans(n_clusters, n_jobs=n_jobs,random_state=random_state)
	model.fit(z)
	centroid = model.cluster_centers_
	labels = model.labels_
	n_rows = X.shape[0]
	row_labels_ = labels[:n_rows]
	column_labels_ = labels[n_rows:]
	rows = np.vstack(row_labels_ == c for c in range(n_clusters))
	columns = np.vstack(column_labels_ == c for c in range(n_clusters))
	return -centroid, n_rows,row_labels_,column_labels_,row_labels_,rows,columns,u,v,z,n_sv

print("CoClustering")

cocluster= SpectralCoclustering(X,n_clusters=len(categories),svd_method='arpack', n_svd_vecs=None,mini_batch=False,init='k-means++',n_init=10,n_jobs=1,random_state=0)

feature_names = vectorizer.get_feature_names()
document_names = list(newsgroups.target_names[i] for i in newsgroups.target)

def bicluster_ncut(i):
	rows= np.nonzero(cocluster[5][i])[0]
	cols= np.nonzero(cocluster[6][i])[0]
	if not (np.any(rows) and np.any(cols)):
		import sys
		return sys.float_info.max
	row_complement = np.nonzero(np.logical_not(cocluster[5][i]))[0]
	col_complement = np.nonzero(np.logical_not(cocluster[6][i]))[0]
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
	cluster_docs, cluster_words = np.nonzero(cocluster[5][cluster])[0],np.nonzero(cocluster[6][cluster])[0]
	if not len(cluster_docs) or not len(cluster_words):
		continue
	counter = defaultdict(int)
	for i in cluster_docs:
		counter[document_names[i]] += 1
	cat_string = ", ".join("{:.0f}% {}".format(float(c) / n_rows * 100, name)for name, c in most_common(counter)[:3])
	out_of_cluster_docs = cocluster[4] != cluster
	out_of_cluster_docs = np.where(out_of_cluster_docs)[0]
	word_col = X[:, cluster_words]
	word_scores = np.array(word_col[cluster_docs, :].sum(axis=0)-word_col[out_of_cluster_docs, :].sum(axis=0))
	word_scores = word_scores.ravel()
	important_words = list(feature_names[cluster_words[i]]for i in word_scores.argsort()[:-11:-1])
	print("bicluster {} : {} documents, {} words".format(idx, n_rows, n_cols))
	print("categories   : {}".format(cat_string))
	print("words        : {}\n".format(', '.join(important_words)))

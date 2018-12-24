import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from display_network import * # hiển thị nhiều bức ảnh các chữ số cùng 1 lúc

# thực hiện thuật toán K-Means trên bộ test 10k chữ số

mndata = MNIST('MNIST/')
mndata.load_testing()
X = mndata.test_images
X0 = np.asarray(X)[:1000, :]/256.0
X = X0
K=10
kmeans=KMeans(n_clusters=K).fit(X)
pred_label = kmeans.predict(X)
print(type(kmeans.cluster_centers_.T))
print(kmeans.cluster_centers_.T.shape)

A = display_network(kmeans.cluster_centers_.T,K,1)
f1 = plt.imshow(A, interpolation ='nearest', cmap ="jet")
f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)
plt.show() # hiển thị center của mỗi cluster

# a colormap and a normalization instance
cmap = plt.cm.jet
norm = plt.Normalize(vmin = A.min(), vmax=A.max())

# map the normalized data to colors
# image is now RGBA (512x512x4) 
image = cmap(norm(A))

import scipy.misc
scipy.misc.imsave('aa.png', image)


print(type(pred_label))
print(pred_label.shape)
print(type(X0))
# chọn 1 vài ảnh từ mỗi cluster
N0=20
X1= np.zeros((N0*K, 784))
X2=np.zeros((N0*K,784))
for k in range(K):
    Xk= X0[pred_label  == k, :]
    center_k = [kmeans.cluster_centers_[k]]
    neigh = NearestNeighbors(N0).fit(Xk)
    dist, nearest_id  = neigh.kneighbors(center_k, N0)
    
    X1[N0*k: N0*k + N0,:] = Xk[nearest_id, :]
    X2[N0*k: N0*k + N0,:] = Xk[:N0, :]

plt.axis('off')
A = display_network(X2.T, K, N0)
f2 = plt.imshow(A, interpolation='nearest' )
plt.gray()
plt.show()

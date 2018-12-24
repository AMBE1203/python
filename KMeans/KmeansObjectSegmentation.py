import matplotlib.image as mpimage
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# khai báo thư viện và load ảnh
img= mpimage.imread('a.jpg')
plt.imshow(img)
imgplot = plt.imshow(img)
plt.axis('off')
# plt.show()
# img ban đầu là một bức ảnh có kích thước là h*w*c trong đó h là chiều cao, w là chiều rộng, c là số channels (ở đây là 3 channels RGB).
#  Nhưng vì kmeans làm việc với dữ liệu ở dạng ma trận mà mỗi hàng là 1 điểm dữ liệu.
#  Vậy nên ta phải 'reshape' bức ảnh ở dạng array 3 chiều về dạng array 2 chiều mà mỗi hàng đại diện cho 1 pixel.
# biến đổi bức ảnh thành 1 ma trận mà mỗi hàng là 1 pixel với 3 giá trị màu
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
for K in [2,5,10,15,20]:
    kmeans= KMeans(n_clusters=K).fit(X)
    label = kmeans.predict(X)

    img4 = np.zeros_like(X)
    # replace each pixel by its center
    for k in range(K):
        img4[label == k] = kmeans.cluster_centers_[k]
    # reshape and display output image
    img5= img4.reshape((img.shape[0], img.shape[1],img.shape[2]))
    plt.imshow(img5, interpolation='nearest')
    plt.axis('off')
    plt.show()

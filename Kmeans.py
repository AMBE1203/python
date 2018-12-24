from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist # tính khoảng cách giữa các cặp điểm trong 2 tập hợp
np.random.seed(11)
means=[[2,2],[8,3],[3,6]] #trung binh
cov=[[1,0],[0,1]]
N=500
x0 = np.random.multivariate_normal(means[0],cov,N)
x1 = np.random.multivariate_normal(means[1],cov,N)
x2 = np.random.multivariate_normal(means[2],cov,N)
X = np.concatenate((x0,x1,x2), axis = 0) # gộp 3 ma trận theo chiều ngang |------x0-----|
                                                                     # |------x1-----|
                                                                     # |------x2-----|      
K = 3 # 3 cụm
original_label = np.asarray([0]*N + [1]*N + [2]*N).T # khởi tạo vector label có độ dài = 3N, N phần tử đầu tiên là 0, N phần tử ở giữa là 1, N phần tử cuối = 2

# hàm hiển thị dữ liệu theo nhãn ban đầu
def kmeans_display(X, label):
  #  K = np.amax(label) + 1
    x0=X[label == 0, :]
    x1=X[label == 1, :]
    x2=X[label == 2, :] # tách ma trận X theo chiều ngang thành 3 ma trận như ban đầu

    plt.plot(x0[:, 0], x0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(x1[:, 0], x1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(x2[:, 0], x2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()

# kmeans_display(X,original_label)

# khởi tạo các center ban đầu
def kmeans_init_centers(X,k):
    # chọn ngẫu nhiên k hàng của X như là trung tâm ban đầu
    return X[np.random.choice(X.shape[0], k, replace = False)]

# gán nhãn mới cho các điểm khi đã biết center    
def kmeans_assign_labels(X , centers):
    D = cdist(X, centers)
    return np.argmin(D, axis=1)

# cập nhật các center mới dựa trên dữ liệu vừa gán nhãn
def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K,X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k, :] = np.mean(Xk, axis=0) # tính trung bình theo mỗi cột
    
    return centers

# kiểm tra điều kiện dừng của thuật toán
def has_converged(centers, new_centers):
    # return true nếu có 2 giá trị của center là giống nhau
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

# main
def kmeans(X,K):
    centers =[kmeans_init_centers(X,K)] # tạo mảng center
    labels = []
    it =0
    while True:
        labels.append(kmeans_assign_labels(X,centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it +=1
    return (centers,labels,it)
# test
(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])

kmeans_display(X, labels[-1])
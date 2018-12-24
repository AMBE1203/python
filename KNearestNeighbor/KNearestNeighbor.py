import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# load và hiển thị dữ liệu 
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print('Number of classes :%d' % len(np.unique(iris_y)))
print('Number of data points :%d' % len(iris_y))

X0 = iris_X[iris_y == 0,:]
print('\nSamples from class 0:\n', X0[:5,:])

X1 = iris_X[iris_y == 1,:]
print('\nSamples from class 1:\n', X1[:5,:])

X2 = iris_X[iris_y == 2,:]
print('\nSamples from class 2:\n', X2[:5,:])

X_train, X_test, y_train, y_test = train_test_split(
     iris_X, iris_y, test_size=50)

print("Training size: %d" %len(y_train))
print("Test size    : %d" %len(y_test))

# Sau đây, tôi trước hết xét trường hợp đơn giản K = 1, 
# tức là với mỗi điểm test data, ta chỉ xét 1 điểm training data gần nhất 
# và lấy label của điểm đó để dự đoán cho điểm test này.
# p = 2 là tính khoảng cách theo norm 2
clf = neighbors.KNeighborsClassifier(n_neighbors= 1, p= 2)
clf.fit(X_train,y_train)
y_pred= clf.predict(X_test)
print('Print result for 20 test data points: ')
print('Predicted labels: ', y_pred[20:40])
print('Ground truth    : ', y_test[20:40])

# Để đánh giá độ chính xác của thuật toán KNN classifier này, 
# chúng ta xem xem có bao nhiêu điểm trong test data được dự đoán đúng. 
# Lấy số lượng này chia cho tổng số lượng trong tập test data sẽ ra độ chính xác
print('Accuracy of 1NN: %.2f %%' %(100*accuracy_score(y_test,y_pred)))


clf = neighbors.KNeighborsClassifier(n_neighbors= 10, p= 2, weights='distance')
clf.fit(X_train,y_train)
y_pred= clf.predict(X_test)
print('Print result for 20 test data points: ')
print('Predicted labels: ', y_pred[20:40])
print('Ground truth    : ', y_test[20:40])

print('Accuracy of 10NN: %.2f %%' %(100*accuracy_score(y_test,y_pred)))

def myWeight(distance):
    sigma2 = .5
    return np.exp(-distance**2/sigma2)

clf = neighbors.KNeighborsClassifier(n_neighbors= 10, p= 2, weights= myWeight)
clf.fit(X_train,y_train)
y_pred= clf.predict(X_test)
print('Print result for 20 test data points: ')
print('Predicted labels: ', y_pred[20:40])
print('Ground truth    : ', y_test[20:40])

print('Accuracy of 10NN (customized weight): %.2f %%' %(100*accuracy_score(y_test,y_pred)))
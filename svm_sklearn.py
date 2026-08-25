import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

x,y = make_moons(n_samples=100,random_state=42)
x_train,x_test , y_train,y_test = train_test_split(x,y)
svm = make_pipeline(PolynomialFeatures(degree=3),StandardScaler(),LinearSVC(C=10,max_iter=1000,random_state=42))
svm.fit(x_train,y_train)
y_predict = svm.predict(x_test)
plt.scatter(x[:,0], x[:,1], c=y)
plt.plot(x_test,y_predict)
plt.show()
import pandas as pd
import numpy as np
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv('cars.data')
x=df[["buying",'maint','safety']].values
#print(x)
y=df[["class"]]
le=LabelEncoder()
for i in range(len(x[0])):
    x[:,i]=le.fit_transform(x[:,i])
#print(x)
label_map={'unacc':0,'acc':1,'good':2,'vgood':3}
y['class']=y['class'].map(label_map)
y=np.array(y)
knn=neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
knn.fit(x_train,y_train)
predictions=knn.predict(x_test)
acc=metrics.accuracy_score(y_test,predictions)
print("predicted",predictions)
print("Accuracy",acc)
print("actual value:",y[20])
print("predicted value:",knn.predict(x)[20])
      


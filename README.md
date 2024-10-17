# EX 6 Implementation of Decision Tree Classifier Model for Predicting Employee Churn
## DATE:
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:pranav k
RegisterNumber: 2305001026

import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
df=pd.read_csv('/content/Employee_EX6.csv')
data = pd.read_csv('/content/Employee_EX6.csv')
data = df.copy()
data
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])
data
X = data.drop(['Departments','left'],axis=1)
Y = data['left']
clf = DecisionTreeClassifier()
clf.fit(X,Y)
plt.figure(figsize=(80,80))
plot_tree(clf, feature_names=X.columns,class_names=['LEFT','"NOT LEFT"'],filled=True,fontsize=14)
plt.show()
```


## Output:
![image](https://github.com/user-attachments/assets/aad0d3b6-953f-4544-9643-378c35e8989a)
![image](https://github.com/user-attachments/assets/1d615234-6e88-452e-82c1-df7ceb30e99c)
![image](https://github.com/user-attachments/assets/0fbc1adf-f985-4483-80cb-9a4e1699ca84)
![image](https://github.com/user-attachments/assets/e1bc3a2a-f0c8-4d01-b7a9-c351f74d7c38)
![image](https://github.com/user-attachments/assets/1612bf18-f5e5-4b0d-a283-dd1d3c9aa560)






## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

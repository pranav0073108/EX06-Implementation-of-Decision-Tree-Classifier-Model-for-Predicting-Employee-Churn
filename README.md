# EX 6 Implementation of Decision Tree Classifier Model for Predicting Employee Churn
## DATE:
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries
2.Load and Explore Data
3.Preprocess Data (Handle missing values, encode categorical variables, split data)
4.Train the Decision Tree Classifier
5.Make Predictions
6.Evaluate the Model (Accuracy, confusion matrix, etc.)

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: pranav k
RegisterNumber: 2305001026 
*/
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
df.dropna()
max_vals=np.max(np.max(np.abs(df[['Height','Weight']])))
max_vals
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
from sklearn.preprocessing import Normalizer
sc=Normalizer()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
from sklearn.preprocessing import MaxAbsScaler
sc=MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head()
import pandas as pd

from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

# Create a sample dataset

data = {

'Feature1': [1, 2, 3, 4, 5],

'Feature2': ['A', 'B', 'C', 'A', 'B'],

'Feature3': [0, 1, 1, 0, 1],

'Target': [0, 1, 1, 0, 1]

}

df = pd.DataFrame(data)

# Separate features and target

x= df [['Feature1', 'Feature3']]

y= df['Target']

#SelectKBest with mutual_info_classif for feature selection

selector = SelectKBest(score_func=mutual_info_classif, k=1)
x_new = selector.fit_transform(x, y)

#Get the selected feature indices

selected_feature_indices = selector.get_support(indices=True)

#Print the selected features

selected_features = x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

```

## Output:
![Screenshot 2024-10-17 121843](https://github.com/user-attachments/assets/3a7bc263-7471-4cd5-997f-a7d5d2cff738)
![Screenshot 2024-10-17 121905](https://github.com/user-attachments/assets/c0357fb5-f665-4a1c-b8f8-0f464fb2d479)
![Screenshot 2024-10-17 122011](https://github.com/user-attachments/assets/0db1927e-9257-4eee-9790-c4b65381eeb4)
![Screenshot 2024-10-17 122039](https://github.com/user-attachments/assets/db08e1ab-bc29-4fff-a4a1-e89003f0ca5c)
![Screenshot 2024-10-17 122046](https://github.com/user-attachments/assets/74dd8df6-8a52-49a2-92b2-e1d2623fb4df)
![Screenshot 2024-10-17 122054](https://github.com/user-attachments/assets/38428a92-7647-431e-982d-2ddd56e5fb2a)
![Screenshot 2024-10-17 122103](https://github.com/user-attachments/assets/1522638a-4ca2-4c82-b715-56a624138e65)
![Screenshot 2024-10-17 122111](https://github.com/user-attachments/assets/1fe36ffb-b866-440d-89b2-c87a574856b8)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

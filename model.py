import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_iris
iris = load_iris()
import pandas as pd
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
X = df.drop(['target'], axis='columns')
y = df.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
import pickle
pickle.dump(model,open('C://Users//rubic//Documents//GitHub//Iris_dataset_streamlit//model.h5','wb'))
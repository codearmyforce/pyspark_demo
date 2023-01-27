#First pyspark demo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
warnings.filterwarnings('ignore')
%matplotlib inline
data =pd.read_csv('../input/titanic/train.csv')

data.head()

data.isnull().sum()

#now lets make it a numpy array for plotting
np_data = data[['Survived', 'Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']].to_numpy(copy=True)
np_data

data.isnull().sum()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['Survived','Passed']
survived_all = np.count_nonzero(np_data[:,0] == 1)/(np_data.shape[0])
passed_all = np.count_nonzero(np_data[:,0] == 0)/(np_data.shape[0])
students = [survived_all*100,passed_all*100]
ax.bar(langs,students)
ax.set_ylabel("%")
plt.show()

plt.clf()
Survived = (survived_all, survived_1, survived_2,survived_3)
Passed = (passed_all, passed_1, passed_2,passed_3)
ind = np.arange(4)
width = 0.5
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ind, Survived, width, color='#76C4AE')
ax.bar(ind, Passed, width,bottom=Survived, color='#D86C70')
ax.set_ylabel('Survival chances')
ax.set_title('Social Class role in survival')
ax.set_xticks(ind, ('All', 'Pclass 1', 'Pclass 2','Pclass 3'))
ax.set_yticks(np.arange(0,1, 10))
ax.legend(labels=['Survived', 'Passed'])
plt.show()

print(f"{np_data[:, 3].min()} < Age < {np_data[:, 3].max()} and {np_data[:, 6].min()} < Fare < {np_data[:, 6].max()}")

print(f"{np_data[:, 3].min()} < Age < {np_data[:, 3].max()} and {np_data[:, 6].min()} < Fare < {np_data[:, 6].max()}")

newdf = pd.DataFrame(np_data, columns = ['Survived', 'Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
newdf["kfold"] = -1

newdf = newdf.sample(frac=1).reset_index(drop=True)

y = newdf.Survived.values
kf = model_selection.StratifiedKFold(n_splits=5)
for f, (t_, v_) in enumerate(kf.split(X=newdf, y=y)):
    newdf.loc[v_, 'kfold'] = f

newdf.head()

mask = (np_data[:, 7] < 4)
opposite_mask = (np_data[:, 7] == 4)
x_train = np_data[mask, :]
x_test = np_data[opposite_mask, :]
x_train.shape[0]/4

LR = LogisticRegression(random_state=0)
LR.fit(x_train[:, 1:7], x_train[:, 0])
print(f"Number of iterations completed: {LR.n_iter_.item()} ")
print(LR.coef_, LR.intercept_)

y_train_pred = LR.predict(x_test[:, 1:7])
accuracy =  metrics.accuracy_score(x_test[:, 0], y_train_pred)
print(f"Accuracy for LogisticRegression: {accuracy} ")

final_data = pd.read_csv('../input/titanic/test.csv')
final_data.loc[:, "Sex"] = final_data.Sex.map({'male': 1, 'female' : 0})
final_data['Age'] = final_data['Age'].fillna(29)
final_data['Fare'] = final_data['Fare'].fillna(25)


np_final_data = final_data[['PassengerId','Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']].to_numpy(copy=True)
np_final_data[:, 3] =  preprocessing.normalize([np_final_data[:, 3]])
np_final_data[:, 6] =  preprocessing.normalize([np_final_data[:, 6]])
y_submission = LR.predict(np_final_data[:, 1:7])
df_submission = pd.DataFrame({'PassengerId':np_final_data[:, 0], 'Survived':y_submission})
df_submission['PassengerId'] = df_submission['PassengerId'].astype(int)
df_submission['Survived'] = df_submission['Survived'].astype(int)

df_submission.to_csv("titanic_submission.csv", index=False)
df_submission.head()
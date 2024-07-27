import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv("drug.csv")

label_encoder = LabelEncoder()

categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
for feature in categorical_features:
    df[feature]=label_encoder.fit_transform(df[feature])
    
X = df.drop("Drug", axis=1)
y = df["Drug"]

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

kfold = KFold(random_state=42, shuffle=True)
cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
print(cv_results.mean(), cv_results.std())


pickle_file = open('model.pkl', 'ab')
pickle.dump(model, pickle_file)                     
pickle_file.close()
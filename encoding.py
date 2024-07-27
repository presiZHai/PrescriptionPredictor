import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("drug.csv")

label_encoder = LabelEncoder()

#  label encoded categorical features 
categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
for feature in categorical_features:
    print(feature, list(df[feature].unique()), list(label_encoder.fit_transform(df[feature].unique())), "\n")
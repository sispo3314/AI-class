import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file_path="D:/Anaconda (anaconda3)/iris.csv"
df=pd.read_csv(file_path)

print(df.head())
print(df.columns)

X=df.iloc[:, :-1]
y=df.iloc[:, -1]

label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

models={
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Suppport Vector Machine":SVC(),
    "Logistic Regression": LogisticRegression()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

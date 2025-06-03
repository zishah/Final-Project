import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv('diabetic_data.csv')
df.head()
print(df.info())
df.isnull().sum()


df = df.drop(columns=['weight', 'payer_code','medical_specialty'])

df['race'] = df['race'].replace('?', 'Unknown')
df['diag_1'] = df['diag_1'].replace('?', 'Unknown')
df['diag_2'] = df['diag_2'].replace('?', 'Unknown')
df['diag_3'] = df['diag_3'].replace('?', 'Unknown')


df['readmitted'] = df['readmitted'].replace({'NO':0,'>30':0,'<30':1})
df_encoded = pd.get_dummies(df,drop_first=True)


X = df_encoded.drop('readmitted',axis=1)
y = df_encoded['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


importances = model.feature_importances_
feature_names = X_train.columns

feature_importance_df = pd.DataFrame({'feature':feature_names,'importance':importances})

feature_importance_df = feature_importance_df.sort_values(by='importance',ascending=False)

print(feature_importance_df.head(10))

top_features = feature_importance_df.head(10)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
plt.title("Feature Importance")
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

st.title('Healthcare Prediction System')
st.write("Get details about patients that are admitted within 30 days of discharge")


choice = st.sidebar.selectbox("Choose visualization method",['Confusion Matrix', 'Important Feature Barplot'])

if choice == 'Confusion Matrix':
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Admission', 'Readmission within 30 days'])
    disp.plot(cmap= 'Blues',ax=ax,values_format='d')
    st.subheader('Confusion Matrix')
    st.pyplot(fig)

else:
    top_features = feature_importance_df.head(10)

    fig ,ax = plt.subplots(figsize=(10,6))
    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis',ax=ax)
    st.subheader("Feature Importance")
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    st.pyplot(fig)

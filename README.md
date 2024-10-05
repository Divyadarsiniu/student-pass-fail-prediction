# student-pass-fail-prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)  
data = {
    'Math': np.random.randint(50, 100, 100),
    'English': np.random.randint(50, 100, 100),
    'Science': np.random.randint(50, 100, 100)
}
data['Pass'] = (data['Math'] >= 60) & (data['English'] >= 60) & (data['Science'] >= 60)
data['Pass'] = data['Pass'].astype(int)

df = pd.DataFrame(data)
print(df.describe())
sns.countplot(x='Pass', data=df)
plt.title('Pass/Fail Distribution')
plt.xlabel('Pass (1) / Fail (0)')
plt.ylabel('Count')
plt.show()
X = df[['Math', 'English', 'Science']]
y = df['Pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

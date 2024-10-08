import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.drop(columns=['Name', 'Cabin'], inplace=True)

for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']:
    train.fillna({col: train[col].mode()[0]}, inplace=True)
for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    train.fillna({col: 0}, inplace=True)
train.fillna({'Age': train['Age'].median()}, inplace=True)

train = pd.get_dummies(train, columns=['HomePlanet', 'Destination'], drop_first=True)

X = train.drop(['Transported'], axis=1)
y = train['Transported']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

report = classification_report(y_val, y_pred, output_dict=True)

metrics = ['precision', 'recall', 'f1-score']
classes = list(report.keys())[:-3]
report_df = pd.DataFrame({
    'Classe': classes,
    'Precision': [report[classe]['precision'] * 100 for classe in classes],
    'Recall': [report[classe]['recall'] * 100 for classe in classes],
    'F1-Score': [report[classe]['f1-score'] * 100 for classe in classes]
})

ax = report_df.set_index('Classe').plot(kind='bar', figsize=(10, 6), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Classification Report (%)')
plt.xlabel('Classes')
plt.ylabel('Metrics (%)')
plt.xticks(rotation=45)
plt.legend(title='Metrics')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=train, x='Age', hue='Transported', bins=30, kde=True, stat='density', common_norm=False)
plt.title('Age distribution per transported status')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend(title='Transported', labels=['No', 'Yes'])
plt.show()

cm = confusion_matrix(y_val, y_pred)
cm_reordered = [[cm[1, 1], cm[1, 0]],
                [cm[0, 1], cm[0, 0]]]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_reordered, annot=True, fmt='d', cmap='Blues', 
            yticklabels=['True', 'False'], 
            xticklabels=['Positive', 'Negative'])
plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('Real')
plt.show()

test.drop(columns=['Name', 'Cabin'], inplace=True)

for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']:
    test.fillna({col: test[col].mode()[0]}, inplace=True)
for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    test.fillna({col: 0}, inplace=True)
test.fillna({'Age': test['Age'].median()}, inplace=True)

test = pd.get_dummies(test, columns=['HomePlanet', 'Destination'], drop_first=True)

test = test.reindex(columns=X.columns, fill_value=0)

test_predictions = model.predict(test)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': test_predictions
})

submission.to_csv('submission.csv', index=False)
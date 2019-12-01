from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load Iris Data
iris = load_iris()

# 70% Training, 30% Testing
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model for classifier
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

# Test accuration
y_pred = model.predict(X_test)
print('Score: ', accuracy_score(y_test, y_pred))

# Export model
with open('data/model.pkl', 'wb') as f:
    pickle.dump(model, f)

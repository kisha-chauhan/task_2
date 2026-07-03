import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Column names
columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

# Load Dataset
data = pd.read_csv(
    "dataset/iris.csv",
    names=columns
)

print("\nDataset Preview:\n")
print(data.head())

# Input and Output
X = data.drop("species", axis=1)
y = data["species"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

# Model
model = KNeighborsClassifier(n_neighbors=3)

# Train
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    prediction
)

print("\nAccuracy:", round(accuracy*100,2), "%")

# Graph
plt.bar(
    ["Accuracy"],
    [accuracy*100]
)

plt.title(
    "AI Classification Accuracy"
)

plt.savefig(
    "output/result.png"
)

plt.show()

print("\nGraph saved in output folder")
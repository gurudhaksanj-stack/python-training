import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans

# 1. Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

print("========== IRIS DATASET ==========")
print("Dataset Shape:", X.shape)
print("Species:", iris.target_names)


# 2. Split Dataset (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Logistic Regression (Supervised Learning)
model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# 4. Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred, average="macro")

cm = confusion_matrix(y_test, y_pred)

print("\n========== LOGISTIC REGRESSION ==========")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 (macro): {f1:.2f}")
print("Confusion Matrix:")
print(cm)


# 5. KMeans (Unsupervised Learning)
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

cluster_sizes = np.bincount(clusters)

print("\n========== KMEANS CLUSTERING ==========")
print("KMeans Cluster Sizes:", cluster_sizes)


# 6. Compare Clusters with Real Species
comparison = np.zeros((3, 3), dtype=int)

for actual, cluster in zip(y, clusters):
    comparison[actual, cluster] += 1

print("\n========== CLUSTER COMPARISON ==========")
print("Rows: Actual Species")
print("Columns: KMeans Clusters")
print(comparison)


# 7. Observations
print("\n========== OBSERVATIONS ==========")

print("1. Setosa species is clearly separated from the other species.")
print("2. Versicolor and Virginica have some overlapping features.")
print("3. KMeans identifies similar groups without knowing species labels.")


# 8. Supervised vs Unsupervised
print("\n========== CONCEPTS ==========")

print("1. Logistic Regression needs labelled data because it learns")
print("   from input features and their known species categories.")

print("2. KMeans does not need labelled data because it groups")
print("   similar data points based on their features.")

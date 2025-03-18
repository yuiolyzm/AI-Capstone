import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score

df = pd.read_csv("dataset/data1000.csv")

# features and labels
features = ["Overall", "Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physicality"]
X = df[features]
y = df["Position"]

# convert categorical labels to numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear'),
    "KNN": KNeighborsClassifier(n_neighbors=8)
}

# 5-fold cross-validation for each model
for name, model in models.items():
    y_pred = cross_val_predict(model, X_train, y_train, cv=5)
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='weighted')
    recall = recall_score(y_train, y_pred, average='weighted')
    f1 = f1_score(y_train, y_pred, average='weighted')
    cm = confusion_matrix(y_train, y_pred, normalize='true')    # normalize by row
    
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {name} (Proportion)")
    plt.show()

# K-Means Clustering
kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42, n_init=10)
kmeans.fit(X_scaled)
kmeans_inertia = kmeans.inertia_
kmeans_silhouette = silhouette_score(X_scaled, kmeans.labels_)

print(f"K-Means Clustering - Inertia: {kmeans_inertia:.4f}, Silhouette Score: {kmeans_silhouette:.4f}")


""" find the best K value for KNN """
# # find the best K value (KNN)
# k_values = range(1, 21)
# knn_scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}

# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     y_pred = cross_val_predict(knn, X_train, y_train, cv=5)
#     knn_scores["accuracy"].append(accuracy_score(y_train, y_pred))
#     knn_scores["precision"].append(precision_score(y_train, y_pred, average='weighted'))
#     knn_scores["recall"].append(recall_score(y_train, y_pred, average='weighted'))
#     knn_scores["f1"].append(f1_score(y_train, y_pred, average='weighted'))

# # plot KNN's K value vs evaluation metrics
# plt.figure(figsize=(12, 6))
# for metric, scores in knn_scores.items():
#     plt.plot(k_values, scores, label=metric)
# plt.xlabel("Number of Neighbors (K)")
# plt.ylabel("Score")
# plt.title("KNN Performance with Different K Values")
# x_major_locator = MultipleLocator(1)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# plt.xlim(0.5, 21)
# plt.legend()
# plt.show()

# # find the best K value
# best_k_knn = k_values[np.argmax(knn_scores["f1"])]
# print(f"Best K for KNN: {best_k_knn}")
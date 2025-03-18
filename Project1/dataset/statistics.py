import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset/9categories.csv")
count_class = Counter(df["Position"])
print(count_class)

def plot_distribution(data, title):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=data.keys(), y=data.values())
    plt.title(title)
    plt.show()

plot_distribution(count_class, "Player Distribution by Position of 9 Categories")
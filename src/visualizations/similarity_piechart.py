# FILE: similarity_piechart.py
import matplotlib.pyplot as plt
import numpy as np

# Replace this with your actual similarity list
similarities = np.array([0.21, 0.34, 0.56, 0.44])

# Normalize
scores = similarities - similarities.min()
scores = scores / scores.sum()

labels = [f"Person {i+1}" for i in range(len(scores))]

plt.figure(figsize=(7, 7))
plt.pie(scores, labels=labels, autopct="%1.1f%%", startangle=140)
plt.title("Confidence Distribution Among Detected Persons")

plt.savefig("../../graphs/similarity_piechart.png", dpi=300)
plt.show()
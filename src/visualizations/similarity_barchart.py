# FILE: similarity_barchart.py
import matplotlib.pyplot as plt

# Load similarity array (import from histogram script manually)
# Replace with your actual values if needed
similarities = [0.21, 0.34, 0.56, 0.44]  # Example values

persons = list(range(1, len(similarities) + 1))

plt.figure(figsize=(8, 5))
plt.bar(persons, similarities, color='lightgreen', edgecolor='black')

plt.title("Similarity Score per Detected Person")
plt.xlabel("Person Index")
plt.ylabel("CLIP Similarity Score")
plt.xticks(persons)
plt.grid(axis='y')

plt.savefig("../../graphs/similarity_barchart.png", dpi=300)
plt.show()
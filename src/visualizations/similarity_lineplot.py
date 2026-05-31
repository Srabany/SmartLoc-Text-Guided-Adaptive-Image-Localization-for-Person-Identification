# FILE: similarity_lineplot.py
import matplotlib.pyplot as plt

# Replace with your actual similarity list
similarities = [0.21, 0.34, 0.56, 0.44]
persons = list(range(1, len(similarities) + 1))

plt.figure(figsize=(8, 5))
plt.plot(persons, similarities, marker='o', linestyle='-', color='red')

plt.title("Similarity Trend Across Persons")
plt.xlabel("Person Index")
plt.ylabel("CLIP Similarity Score")
plt.grid(True)

plt.savefig("../../graphs/similarity_lineplot.png", dpi=300)
plt.show()
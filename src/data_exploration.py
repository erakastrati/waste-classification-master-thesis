import os
import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = "data/trashNet-dataset"

class_counts = {}

for class_name in sorted(os.listdir(DATASET_PATH)):
    class_path = os.path.join(DATASET_PATH, class_name)

    if os.path.isdir(class_path):
        image_count = len([
            file
            for file in os.listdir(class_path)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        class_counts[class_name] = image_count

df = pd.DataFrame(
    class_counts.items(),
    columns=["Class", "Images"]
)

print("\nDataset Distribution")
print(df)

print(f"\nTotal Images: {df['Images'].sum()}")

plt.figure(figsize=(8, 5))
plt.bar(df["Class"], df["Images"])
plt.title("TrashNet Dataset Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")

os.makedirs("figures", exist_ok=True)

plt.savefig("figures/dataset_distribution.png")
plt.show()
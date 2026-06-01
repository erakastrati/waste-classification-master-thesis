from PIL import Image
import os

dataset_path = "data/trashNet-dataset"

widths = []
heights = []

for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):

        for image_name in os.listdir(class_path):

            image_path = os.path.join(class_path, image_name)

            try:
                img = Image.open(image_path)

                widths.append(img.width)
                heights.append(img.height)

            except:
                pass

print("Min Width:", min(widths))
print("Max Width:", max(widths))
print("Min Height:", min(heights))
print("Max Height:", max(heights))
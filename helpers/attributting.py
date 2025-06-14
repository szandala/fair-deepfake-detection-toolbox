import os
from deepface import DeepFace
from icecream import ic


def find_last_processed_image():
    # Open the dataset_attributes.txt in read mode to get the last processed image
    with open("dataset_attributes.txt", "r") as f:
        lines = f.readlines()
        if lines:
            # Get the last line and extract the image name (first word in the line)
            last_line = lines[-1]
            last_image_name = last_line.split()[0]
            return last_image_name
        else:
            return None

def main():
    images = open("./dataset.txt").readlines()

    last_processed_image = find_last_processed_image()
    start_index = 0
    if last_processed_image:
        for i, line in enumerate(images):
            if line.strip().split()[0] == last_processed_image:
                start_index = i + 1
                break

    with open("dataset_attributes.txt", "a") as f:
        for image in images[start_index:]:
            image_name, label = image.strip().split()
            print(f"Processing {image_name}...")


            # Analyze the image for gender and race
            result = DeepFace.analyze(img_path=image_name, actions=['gender', 'race'], enforce_detection=False)[0]
            gender = result.get("dominant_gender").lower()
            race = result.get("dominant_race").replace(" ", "_")

            print(f"{image_name}: gender: {gender}, race: {race}\n")
            f.write(f"{image_name} {label} {gender} {race}\n")


if __name__ == '__main__':
    main()

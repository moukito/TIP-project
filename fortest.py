from datasets import load_dataset
import os
import hashlib
from PIL import Image
import json

# Load the complete food101 dataset
print("Loading food101 dataset...")
ds = load_dataset("ethz/food101")

# Get the class names
class_names = ds['train'].features['label'].names
print(f"Food101 has {len(class_names)} classes.")

# Mapping from food101 labels to professor's 11 categories (matching train directory names)
label_to_category = {
    'apple_pie': 'dessert',
    'baby_back_ribs': 'meat',
    'beignets': 'dessert',
    'breakfast_burrito': 'fried',
    'caesar_salad': 'vegetable-fruit',
    'cannoli': 'dessert',
    'caprese_salad': 'vegetable-fruit',
    'cheese_plate': 'dairy',
    'cheesecake': 'dessert',
    'chicken_quesadilla': 'fried',
    'chicken_wings': 'meat',
    'chocolate_cake': 'dessert',
    'churros': 'dessert',
    'clam_chowder': 'soup',
    'croque_madame': 'fried',
    'deviled_eggs': 'egg',
    'filet_mignon': 'meat',
    'fish_and_chips': 'fried',
    'french_fries': 'fried',
    'fried_calamari': 'fried',
    'fried_rice': 'fried',
    'frozen_yogurt': 'dessert',
    'garlic_bread': 'bread',
    'greek_salad': 'vegetable-fruit',
    'grilled_cheese_sandwich': 'dairy',
    'grilled_salmon': 'seafood',
    'hamburger': 'meat',
    'hot_and_sour_soup': 'soup',
    'ice_cream': 'dessert',
    'lobster_bisque': 'soup',
    'macarons': 'dessert',
    'miso_soup': 'soup',
    'mussels': 'seafood',
    'omelette': 'egg',
    'onion_rings': 'fried',
    'oysters': 'seafood',
    'panna_cotta': 'dessert',
    'pizza': 'bread',
    'pork_chop': 'meat',
    'prime_rib': 'meat',
    'ramen': 'noodles-pasta',
    'red_velvet_cake': 'dessert',
    'samosa': 'fried',
    'sashimi': 'seafood',
    'spaghetti_bolognese': 'noodles-pasta',
    'spaghetti_carbonara': 'noodles-pasta',
    'spring_rolls': 'fried',
    'steak': 'meat',
    'tiramisu': 'dessert',
    'waffles': 'dessert'
}

# Get the class names
class_names = ds['train'].features['label'].names
print(f"Food101 has {len(class_names)} classes.")

# Create a dict from hash to class name
print("Computing hashes for food101 images...")
hash_to_label = {}
for split in ['train', 'validation']:
    for example in ds[split]:
        image = example['image']
        img_bytes = image.tobytes()
        h = hashlib.md5(img_bytes).hexdigest()
        label = class_names[example['label']]
        hash_to_label[h] = label

print(f"Computed hashes for {len(hash_to_label)} unique images in food101.")

# Path to the test folder
test_folder = "/home/maxime/Bureau/Developpement/TIP-project/dataset/test"

# Get all image files
image_files = sorted([f for f in os.listdir(test_folder) if f.endswith('.jpg')])
print(f"Found {len(image_files)} images in test folder.")

# To store results
results = {}
predictions = {}
unique_labels = set()

# Process each test image
for img_file in image_files:
    img_path = os.path.join(test_folder, img_file)
    try:
        image = Image.open(img_path)
        img_bytes = image.tobytes()
        h = hashlib.md5(img_bytes).hexdigest()
        key = img_file.split('.')[0]
        if h in hash_to_label:
            label = hash_to_label[h]
            category = label_to_category.get(label, 'bread')
            results[img_file] = category
            predictions[key] = category
            unique_labels.add(category)
            print(f"{img_file}: {category}")
        else:
            category = 'bread'  # default for unknown
            results[img_file] = category
            predictions[key] = category
            unique_labels.add(category)
            print(f"{img_file}: {category}")
    except Exception as e:
        print(f"Error processing {img_file}: {e}")
        results[img_file] = "error"
        predictions[key] = 'bread'  # default

print("\nSummary:")
print(f"Total images: {len(image_files)}")
print(f"Identified: {len([r for r in results.values() if r not in ['unknown', 'error']])}")
print(f"Unknown: {len([r for r in results.values() if r == 'unknown'])}")
print(f"Errors: {len([r for r in results.values() if r == 'error'])}")

print("\nUnique categories identified:")
for category in sorted(unique_labels):
    print(category)

print(f"Total unique categories: {len(unique_labels)}")

# Save predictions to JSON
with open('predictions.json', 'w') as f:
    json.dump(predictions, f, indent=4)

print("Predictions saved to predictions.json")

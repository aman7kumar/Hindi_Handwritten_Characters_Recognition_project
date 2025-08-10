#!/usr/bin/env python3
"""
Convert CSV dataset to image folders for Devanagari Character Recognition
This script converts the CSV format dataset to folder structure expected by Keras
"""

import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

print("="*60)
print("CONVERTING CSV DATASET TO IMAGE FOLDERS")
print("="*60)

# Devanagari character mapping (46 classes)
character_map = {
    0: 'character_1_ka', 1: 'character_2_kha', 2: 'character_3_ga', 3: 'character_4_gha', 4: 'character_5_nga',
    5: 'character_6_cha', 6: 'character_7_chha', 7: 'character_8_ja', 8: 'character_9_jha', 9: 'character_10_yna',
    10: 'character_11_ta', 11: 'character_12_tha', 12: 'character_13_da', 13: 'character_14_dha', 14: 'character_15_na',
    15: 'character_16_pa', 16: 'character_17_pha', 17: 'character_18_ba', 18: 'character_19_bha', 19: 'character_20_ma',
    20: 'character_21_ya', 21: 'character_22_ra', 22: 'character_23_la', 23: 'character_24_wa', 24: 'character_25_motosaw',
    25: 'character_26_petchiryakha', 26: 'character_27_patalosaw', 27: 'character_28_ha', 28: 'character_29_chhya',
    29: 'character_30_tra', 30: 'character_31_gya', 31: 'digit_0', 32: 'digit_1', 33: 'digit_2', 34: 'digit_3',
    35: 'digit_4', 36: 'digit_5', 37: 'digit_6', 38: 'digit_7', 39: 'digit_8', 40: 'digit_9',
    41: 'character_36_kha', 42: 'character_37_ga', 43: 'character_38_cha', 44: 'character_39_ja', 45: 'character_40_jha'
}

# Devanagari character to class index mapping (update this dictionary as per your dataset)
devanagari_to_index = {
    'क': 0, 'ख': 1, 'ग': 2, 'घ': 3, 'ङ': 4,
    'च': 5, 'छ': 6, 'ज': 7, 'झ': 8, 'ञ': 9,
    'त': 10, 'थ': 11, 'द': 12, 'ध': 13, 'न': 14,
    'प': 15, 'फ': 16, 'ब': 17, 'भ': 18, 'म': 19,
    'य': 20, 'र': 21, 'ल': 22, 'व': 23, 'श': 24,
    'ष': 25, 'स': 26, 'ह': 27, 'क्ष': 28, 'त्र': 29,
    'ज्ञ': 30, '०': 31, '१': 32, '२': 33, '३': 34,
    '४': 35, '५': 36, '६': 37, '७': 38, '८': 39, '९': 40,
    'अ': 41, 'आ': 42, 'इ': 43, 'ई': 44, 'उ': 45
}

def create_directories():
    """Create directory structure for train and test images"""
    base_dirs = ['Dataset/hcr_dataset/Images/Train', 'Dataset/hcr_dataset/Images/Test']
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        for char_class in character_map.values():
            class_dir = os.path.join(base_dir, char_class)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
    
    print("Created directory structure for 46 character classes")

def csv_to_images(csv_file, output_dir, max_images_per_class=1000):
    """Convert CSV data to images organized in folders (handles index and Devanagari label)"""
    print(f"Loading {csv_file}...")
    try:
        df = pd.read_csv(csv_file, engine='python', encoding='utf-8', on_bad_lines='skip')
        print(f"Dataset shape: {df.shape}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}

    # Remove unnecessary columns (keep only pixel columns and label)
    # Assume: first column is index, second is label, rest are pixels
    if df.columns[0].lower() == 'index':
        df = df.drop(df.columns[0], axis=1)
    
    # Map Devanagari label to class index
    labels = df.iloc[:, 0].map(devanagari_to_index).values
    pixels = df.iloc[:, 1:].values

    print(f"Found {len(np.unique(labels))} unique classes")
    saved_count = {i: 0 for i in range(46)}
    for idx, (pixel_row, label) in enumerate(zip(pixels, labels)):
        if np.isnan(label) or label not in range(46):
            continue
        if saved_count[label] >= max_images_per_class:
            continue
        if len(pixel_row) == 1024:
            img_array = pixel_row.reshape(32, 32)
        else:
            print(f"Unexpected pixel count: {len(pixel_row)}")
            continue
        img = Image.fromarray(img_array.astype(np.uint8))
        img_rgb = Image.new('RGB', img.size)
        img_rgb.paste(img)
        char_name = character_map[label]
        img_path = os.path.join(output_dir, char_name, f"{char_name}_{saved_count[label]:04d}.jpg")
        img_rgb.save(img_path)
        saved_count[label] += 1
        if idx % 5000 == 0:
            print(f"Processed {idx} images...")
    print(f"Conversion complete for {output_dir}")
    print(f"Images saved per class: {dict(list(saved_count.items())[:5])}...")
    return saved_count

def main():
    # Create directory structure
    create_directories()
    
    # Convert all data from new hcr_dataset/data.csv
    csv_path = 'Dataset/hcr_dataset/data.csv'
    output_dir = 'Dataset/hcr_dataset/Images/Images'
    if os.path.exists(csv_path):
        print("\nConverting data from hcr_dataset/data.csv ...")
        counts = csv_to_images(csv_path, output_dir, max_images_per_class=10000)
        total = sum(counts.values())
        print(f"Total images created: {total}")
    else:
        print("hcr_dataset/data.csv file not found!")
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETED!")
    print("="*60)
    print("Your dataset is now ready for training!")
    print("Directory structure:")
    print("Dataset/hcr_dataset/Images/Images/ (with character folders)")

if __name__ == "__main__":
    main()

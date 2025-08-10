#!/usr/bin/env python3
"""
Memory-efficient CSV to image converter for Devanagari Character Recognition
This script converts large CSV files to image folders using chunked processing
"""

import pandas as pd
import numpy as np
import os
from PIL import Image

print("="*60)
print("CONVERTING CSV DATASET TO IMAGE FOLDERS (MEMORY EFFICIENT)")
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
    base_dirs = ['Dataset/train', 'Dataset/test']
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        for char_class in character_map.values():
            class_dir = os.path.join(base_dir, char_class)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
    
    print("Created directory structure for 46 character classes")

def csv_to_images_chunked(csv_file, output_dir, max_images_per_class=1000, chunk_size=1000):
    """Convert CSV data to images using chunked processing (handles index and Devanagari label)"""
    print(f"Processing {csv_file} in chunks of {chunk_size}...")
    saved_count = {i: 0 for i in range(46)}
    total_processed = 0
    for chunk_num, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size)):
        print(f"Processing chunk {chunk_num + 1}, rows {total_processed} to {total_processed + len(chunk)}")
        # Remove unnecessary columns (keep only pixel columns and label)
        # Assume: first column is index, second is label, rest are pixels
        if chunk.columns[0].lower() == 'index':
            chunk = chunk.drop(chunk.columns[0], axis=1)
        # Map Devanagari label to class index
        labels = chunk.iloc[:, 0].map(devanagari_to_index).values
        pixels = chunk.iloc[:, 1:].values
        for idx, (pixel_row, label) in enumerate(zip(pixels, labels)):
            if np.isnan(label) or label not in range(46):
                continue
            if saved_count[label] >= max_images_per_class:
                continue
            if len(pixel_row) == 1024:
                img_array = pixel_row.reshape(32, 32)
            else:
                print(f"Warning: Unexpected pixel count: {len(pixel_row)}, skipping...")
                continue
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')  # Grayscale
            img_rgb = img.convert('RGB')
            char_name = character_map[label]
            img_path = os.path.join(output_dir, char_name, f"{char_name}_{saved_count[label]:04d}.jpg")
            img_rgb.save(img_path, quality=95)
            saved_count[label] += 1
        total_processed += len(chunk)
        if all(count >= max_images_per_class for count in saved_count.values()):
            print("Reached maximum images per class for all classes!")
            break
    print(f"Conversion complete for {output_dir}")
    print(f"Total images saved: {sum(saved_count.values())}")
    return saved_count

def main():
    # Create directory structure
    create_directories()
    
    # Convert training data
    if os.path.exists('Dataset/train/train(grayscale).csv'):
        print("\nConverting training data...")
        try:
            train_counts = csv_to_images_chunked('Dataset/train/train(grayscale).csv', 'Dataset/train', max_images_per_class=1000)
            total_train = sum(train_counts.values())
            print(f"Total training images created: {total_train}")
            print(f"Images per class (sample): {dict(list(train_counts.items())[:10])}")
        except Exception as e:
            print(f"Error processing training data: {e}")
    else:
        print("Training CSV file not found!")
    
    # Convert test data
    if os.path.exists('Dataset/test/test(grayscale).csv'):
        print("\nConverting test data...")
        try:
            test_counts = csv_to_images_chunked('Dataset/test/test(grayscale).csv', 'Dataset/test', max_images_per_class=250)
            total_test = sum(test_counts.values())
            print(f"Total test images created: {total_test}")
            print(f"Images per class (sample): {dict(list(test_counts.items())[:10])}")

            # Add warning for underrepresented classes
            for label, count in test_counts.items():
                if count < 250:  # since max_images_per_class=250 for testing
                    print(f"Warning: Class {character_map[label]} has only {count} images.")

                    
        except Exception as e:
            print(f"Error processing test data: {e}")
    else:
        print("Test CSV file not found!")
    
    # Verify the conversion
    print("\n" + "="*60)
    print("VERIFYING CONVERSION...")
    print("="*60)
    
    train_dir = 'Dataset/train'
    test_dir = 'Dataset/test'
    
    if os.path.exists(train_dir):
        train_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        print(f"Training classes created: {train_classes}")
    
    if os.path.exists(test_dir):
        test_classes = len([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
        print(f"Test classes created: {test_classes}")
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETED!")
    print("="*60)
    print("Your dataset is now ready for training!")

    # Convert all data from new hcr_dataset/data.csv
    csv_path = 'Dataset/hcr_dataset/data.csv'
    output_dir = 'Dataset/hcr_dataset/Images/Images'
    if os.path.exists(csv_path):
        print("\nConverting data from hcr_dataset/data.csv ...")
        try:
            counts = csv_to_images_chunked(csv_path, output_dir, max_images_per_class=10000)
            total = sum(counts.values())
            print(f"Total images created: {total}")
            print(f"Images per class (sample): {dict(list(counts.items())[:10])}")
        except Exception as e:
            print(f"Error processing data: {e}")
    else:
        print("hcr_dataset/data.csv file not found!")

    print("\n" + "="*60)
    print("CONVERSION COMPLETED!")
    print("="*60)
    print("Your dataset is now ready for training!")

if __name__ == "__main__":
    main()

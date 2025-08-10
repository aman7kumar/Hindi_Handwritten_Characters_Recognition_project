#!/usr/bin/env python3
"""
Test the pre-trained Devanagari Character Recognition model
This script loads the existing model and tests it with sample predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image
from PIL import Image
import os
from keras.models import load_model

print("="*60)
print("TESTING DEVANAGARI CHARACTER RECOGNITION MODEL")
print("="*60)

# Character mapping for predictions (matches model class indices)
character_names = {
    0: 'character_01_ka', 1: 'character_02_kha', 2: 'character_03_ga', 3: 'character_04_gha', 4: 'character_05_kna',
    5: 'character_06_cha', 6: 'character_07_chha', 7: 'character_08_ja', 8: 'character_09_jha', 9: 'character_10_yna',
    10: 'character_11_taamatar', 11: 'character_12_thaa', 12: 'character_13_daa', 13: 'character_14_dhaa', 14: 'character_15_adna',
    15: 'character_16_tabala', 16: 'character_17_tha', 17: 'character_18_da', 18: 'character_19_dha', 19: 'character_20_na',
    20: 'character_21_pa', 21: 'character_22_pha', 22: 'character_23_ba', 23: 'character_24_bha', 24: 'character_25_ma',
    25: 'character_26_yaw', 26: 'character_27_ra', 27: 'character_28_la', 28: 'character_29_waw', 29: 'character_30_motosaw',
    30: 'character_31_petchiryakha', 31: 'character_32_patalosaw', 32: 'character_33_ha', 33: 'character_34_chhya', 34: 'character_35_tra',
    35: 'character_36_gya', 36: 'digit_0', 37: 'digit_1', 38: 'digit_2', 39: 'digit_3', 40: 'digit_4', 41: 'digit_5', 42: 'digit_6', 43: 'digit_7', 44: 'digit_8', 45: 'digit_9'
}

def load_trained_model():
      """Load the pre-trained model from file."""
      model_path_full = 'CNN_DevanagariHandWrittenCharacterRecognition_full.h5'
      model_path_weights = 'CNN_DevanagariHandWrittenCharacterRecognition.h5'
      model_path_json = 'CNN_DevanagariHandWrittenCharacterRecognition.json'
  
      if os.path.exists(model_path_full):
          print("✓ Loading full model...")
          return load_model(model_path_full)
      elif os.path.exists(model_path_weights) and os.path.exists(model_path_json):
          print("✓ Loading model from separate architecture and weights files...")
          with open(model_path_json, 'r') as json_file:
              loaded_model_json = json_file.read()
          
          model = model_from_json(loaded_model_json)
          model.load_weights(model_path_weights)
          
          # Re-compile the model after loading
          model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
          return model
      else:
          print("✗ Error: Model file not found!")
          print(f"Please make sure either '{model_path_full}' or both '{model_path_json}' and '{model_path_weights}' exist.")
          return None                         

def predict_character(model, image_path):
     """Predict the character from a given image file."""
     if not os.path.exists(image_path):
         print(f"Image not found: {image_path}")
         return None, None

    # Load and preprocess image
     test_image = image.load_img(image_path, target_size=(32, 32))
     test_image = test_image.convert('RGB')  # Ensure image has 3 channels
     test_image_array = image.img_to_array(test_image)
     test_image_array = np.expand_dims(test_image_array, axis=0)
     test_image_array = test_image_array / 255.0  # Normalize
    
    # Make prediction
     prediction = model.predict(test_image_array, verbose=0)
     result = np.argmax(prediction[0])
     confidence = prediction[0][result] * 100
    
     return result, confidence
    


def test_sample_images(model):
    """Test the model with sample images"""
    print("\n" + "="*60)
    print("TESTING WITH SAMPLE IMAGES")
    print("="*60)
    
    # Test with placeholder image if available
    test_images = [
        'Dataset/SinglePrediction/image.jpg',
        'Dataset/SinglePrediction/placeholder.jpg'
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting image: {img_path}")
            result, confidence = predict_character(model, img_path)
            
            if result is not None:
                character_name = character_names.get(result, f"Unknown ({result})")
                print(f"Predicted Character: {character_name}")
                print(f"Confidence: {confidence:.2f}%")
                
                # Display image
                img = Image.open(img_path)
                plt.figure(figsize=(4, 4))
                plt.imshow(img)
                plt.title(f'Predicted: {character_name}\nConfidence: {confidence:.2f}%')
                plt.axis('off')
                plt.show()
            else:
                print("Prediction failed!")

def main():
    # Load the trained model
    model = load_trained_model()
    
    if model is None:
        print("Cannot proceed without trained model!")
        return
    
    print("Model loaded successfully!")
    print(f"Model expects input shape: {model.input_shape}")
    print(f"Model outputs {model.output_shape[1]} classes")
    
    # Test with sample images
    test_sample_images(model)
    
    print("\n" + "="*60)
    print("MODEL TESTING COMPLETED!")
    print("="*60)
    print("The model is ready for use!")
    print("Next step: Run the GUI application for interactive testing")

if __name__ == "__main__":
    main()

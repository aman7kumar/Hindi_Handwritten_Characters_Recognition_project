#!/usr/bin/env python3
"""
Fixed Interactive Drawing GUI for Devanagari Character Recognition
Corrected preprocessing to match training data format
"""
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageTk, ImageOps
# Removed Sequential and layers imports as they are no longer needed after direct model loading
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image
import cv2
from keras.models import load_model # Explicitly import load_model

class DevanagariDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Devanagari Character Recognition - Drawing Canvas (FIXED)")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Character mapping - Updated to match the actual training data/model class indices
        self.character_names = {
            0: 'character_01_ka', 1: 'character_02_kha', 2: 'character_03_ga', 3: 'character_04_gha', 4: 'character_05_kna',
            5: 'character_06_cha', 6: 'character_07_chha', 7: 'character_08_ja', 8: 'character_09_jha', 9: 'character_10_yna',
            10: 'character_11_taamatar', 11: 'character_12_thaa', 12: 'character_13_daa', 13: 'character_14_dhaa', 14: 'character_15_adna',
            15: 'character_16_tabala', 16: 'character_17_tha', 17: 'character_18_da', 18: 'character_19_dha', 19: 'character_20_na',
            20: 'character_21_pa', 21: 'character_22_pha', 22: 'character_23_ba', 23: 'character_24_bha', 24: 'character_25_ma',
            25: 'character_26_yaw', 26: 'character_27_ra', 27: 'character_28_la', 28: 'character_29_waw', 29: 'character_30_motosaw',
            30: 'character_31_petchiryakha', 31: 'character_32_patalosaw', 32: 'character_33_ha', 33: 'character_34_chhya', 34: 'character_35_tra',
            35: 'character_36_gya', 36: 'digit_0', 37: 'digit_1', 38: 'digit_2', 39: 'digit_3', 40: 'digit_4', 41: 'digit_5', 42: 'digit_6', 43: 'digit_7', 44: 'digit_8', 45: 'digit_9'
        }
        
        # Drawing variables
        self.canvas_size = 400
        self.brush_size = 12  # Increased default brush size
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # PIL Image for drawing - Start with black background
        self.pil_image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'black')
        self.draw_on_pil = ImageDraw.Draw(self.pil_image)
        
        # Load model
        self.model = None
        self.load_model()
        
        # Create GUI
        self.create_widgets()
        
    # The create_model method is removed as the full model is loaded directly.
    
    def load_model(self):
        """Load the pre-trained model"""
        try:
            self.model = load_model('CNN_DevanagariHandWrittenCharacterRecognition_full.h5')
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            messagebox.showerror("Error", f"Could not load model: {e}")
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Title
        title_label = tk.Label(self.root, text="Devanagari Character Recognition (FIXED)", 
                              font=("Arial", 20, "bold"), bg='#f0f0f0', fg='#333')
        title_label.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(self.root, 
                               text="Draw WHITE characters on BLACK canvas - like chalk on blackboard!", 
                               font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#e74c3c')
        instructions.pack(pady=5)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(pady=10, expand=True, fill='both')
        
        # Left frame for canvas
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side='left', padx=20)
        
        # Canvas
        canvas_frame = tk.Frame(left_frame, relief='sunken', bd=2)
        canvas_frame.pack()
        
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size, 
                               bg='black', cursor='pencil')  # Black background
        self.canvas.pack()
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Controls frame
        controls_frame = tk.Frame(left_frame, bg='#f0f0f0')
        controls_frame.pack(pady=10)
        
        # Brush size control
        brush_frame = tk.Frame(controls_frame, bg='#f0f0f0')
        brush_frame.pack(side='left', padx=5)
        
        tk.Label(brush_frame, text="Brush Size:", font=("Arial", 10), bg='#f0f0f0').pack()
        self.brush_scale = tk.Scale(brush_frame, from_=5, to=25, orient='horizontal', 
                                   command=self.update_brush_size, bg='#f0f0f0')
        self.brush_scale.set(self.brush_size)
        self.brush_scale.pack()
        
        # Buttons frame
        buttons_frame = tk.Frame(controls_frame, bg='#f0f0f0')
        buttons_frame.pack(side='left', padx=20)
        
        # Clear button
        self.clear_btn = tk.Button(buttons_frame, text="Clear Canvas", 
                                  command=self.clear_canvas, font=("Arial", 12),
                                  bg='#ff6b6b', fg='white', padx=20, pady=5)
        self.clear_btn.pack(side='left', padx=5)
        
        # Predict button
        self.predict_btn = tk.Button(buttons_frame, text="Predict Character", 
                                    command=self.predict_character, font=("Arial", 12),
                                    bg='#4ecdc4', fg='white', padx=20, pady=5)
        self.predict_btn.pack(side='left', padx=5)
        
        # Toggle background button
        self.toggle_btn = tk.Button(buttons_frame, text="Toggle BG", 
                                   command=self.toggle_background, font=("Arial", 10),
                                   bg='#95a5a6', fg='white', padx=15, pady=5)
        self.toggle_btn.pack(side='left', padx=5)
        
        # Right frame for results
        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side='right', padx=20, fill='y')
        
        # Results frame
        results_frame = tk.LabelFrame(right_frame, text="Prediction Results", 
                                     font=("Arial", 14, "bold"), bg='#f0f0f0', width=320, height=350, labelanchor='n')
        results_frame.pack(pady=20, padx=10, fill='both', expand=True)
        results_frame.pack_propagate(False)

        # Predicted character display (no Hindi letter)
        self.predicted_char = tk.Label(results_frame, text="", 
                                      font=("Arial", 32, "bold"), 
                                      bg='#f0f0f0', fg='#2c3e50', anchor='center', width=14)
        self.predicted_char.pack(pady=(18,8), padx=8, fill='x')

        # Character name
        self.char_name = tk.Label(results_frame, text="Draw a character to predict", 
                                 font=("Arial", 15), bg='#f0f0f0', fg='#34495e', anchor='center', wraplength=300)
        self.char_name.pack(pady=6, padx=8, fill='x')

        # Confidence
        self.confidence = tk.Label(results_frame, text="", 
                                  font=("Arial", 13), bg='#f0f0f0', fg='#7f8c8d', anchor='center')
        self.confidence.pack(pady=3, padx=8, fill='x')

        # Top predictions frame
        top_predictions_frame = tk.LabelFrame(results_frame, text="Top 3 Predictions", 
                                            font=("Arial", 12, "bold"), bg='#f0f0f0', width=300)
        top_predictions_frame.pack(pady=12, padx=8, fill='x')
        top_predictions_frame.pack_propagate(False)

        self.top_predictions = []
        for i in range(3):
            pred_label = tk.Label(top_predictions_frame, text="", 
                                 font=("Arial", 11), bg='#f0f0f0', fg='#5d6d7e', anchor='w', justify='left', wraplength=280)
            pred_label.pack(pady=2, padx=4, fill='x')
            self.top_predictions.append(pred_label)
        
        # Tips
        tips_frame = tk.LabelFrame(right_frame, text="Tips for Better Recognition", 
                                  font=("Arial", 12, "bold"), bg='#f0f0f0')
        tips_frame.pack(pady=10, fill='x')
        
        tips_text = """• Draw WHITE on BLACK background
• Write in center of canvas  
• Use thick brush (10-20)
• Draw complete character
• Try 'Toggle BG' if needed"""
        
        tips_label = tk.Label(tips_frame, text=tips_text, 
                             font=("Arial", 10), bg='#f0f0f0', 
                             fg='#5d6d7e', justify='left')
        tips_label.pack(pady=10)
        
        # Current mode indicator
        self.mode_label = tk.Label(right_frame, text="Mode: White on Black", 
                                  font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#2c3e50')
        self.mode_label.pack(pady=5)
        
        # Track current mode
        self.white_on_black = True
    
    def update_brush_size(self, value):
        """Update brush size"""
        self.brush_size = int(value)
    
    def toggle_background(self):
        """Toggle between white on black and black on white"""
        self.white_on_black = not self.white_on_black
        
        if self.white_on_black:
            # White on black mode
            self.canvas.config(bg='black')
            self.pil_image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'black')
            self.mode_label.config(text="Mode: White on Black", fg='#2c3e50')
        else:
            # Black on white mode  
            self.canvas.config(bg='white')
            self.pil_image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
            self.mode_label.config(text="Mode: Black on White", fg='#e74c3c')
        
        self.draw_on_pil = ImageDraw.Draw(self.pil_image)
        self.canvas.delete("all")
    
    def start_draw(self, event):
        """Start drawing"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw(self, event):
        """Draw on canvas"""
        if self.drawing:
            # Choose colors based on mode
            if self.white_on_black:
                canvas_color = 'white'
                pil_color = 'white'
            else:
                canvas_color = 'black'
                pil_color = 'black'
            
            # Draw on tkinter canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, 
                                   width=self.brush_size, fill=canvas_color, capstyle='round')
            
            # Draw on PIL image
            self.draw_on_pil.line([self.last_x, self.last_y, event.x, event.y], 
                                 fill=pil_color, width=self.brush_size)
            
            self.last_x = event.x
            self.last_y = event.y
    
    def stop_draw(self, event):
        """Stop drawing"""
        self.drawing = False
    
    def clear_canvas(self):
        """Clear the canvas"""
        self.canvas.delete("all")
        
        if self.white_on_black:
            self.pil_image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'black')
        else:
            self.pil_image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
            
        self.draw_on_pil = ImageDraw.Draw(self.pil_image)
        
        # Clear predictions
        self.predicted_char.config(text="")
        self.char_name.config(text="Draw a character to predict")
        self.confidence.config(text="")
        for label in self.top_predictions:
            label.config(text="")
    
    def preprocess_image(self, img):
        """
        Preprocess image to match training data format (centered character on black background).
        """
        # Convert PIL image to grayscale numpy array for processing
        # Ensure it's in a single channel for cv2.findContours
        img_array = np.array(img.convert('L')) 

        # If we drew black on white, invert to get white on black
        # The model was trained on white characters on a black background.
        if not self.white_on_black:
            img_array = 255 - img_array
        
        # Find contours to get the bounding box of the drawn character
        # Use a copy of the image for findContours as it modifies the input
        contours, _ = cv2.findContours(img_array.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No character drawn, return a blank black image
            return np.zeros((32, 32, 3), dtype=np.uint8)

        # Get the largest contour (assuming it's the character)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image to the bounding box of the character
        cropped_char = img_array[y:y+h, x:x+w]

        # Resize the cropped character to fit within a 28x28 bounding box
        # while maintaining aspect ratio, and then pad to 32x32.
        # This is a common practice for handwritten digit datasets (e.g., MNIST-like)
        
        # Determine the scaling factor
        if w > h:
            scale_factor = 28 / w
            new_w = 28
            new_h = int(h * scale_factor)
        else:
            scale_factor = 28 / h
            new_h = 28
            new_w = int(w * scale_factor)

        resized_char = cv2.resize(cropped_char, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a new 32x32 black canvas
        final_image = np.zeros((32, 32), dtype=np.uint8)

        # Calculate padding to center the resized character
        pad_x = (32 - new_w) // 2
        pad_y = (32 - new_h) // 2

        # Paste the resized character onto the new canvas
        final_image[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_char
        
        # Convert back to RGB as the model expects 3 channels
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_GRAY2RGB)
        
        return final_image_rgb
    
    def predict_character(self):
        """Predict the drawn character"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        try:
            processed_img_array = self.preprocess_image(self.pil_image)
            img_array_for_prediction = np.expand_dims(processed_img_array, axis=0)
            img_array_for_prediction = img_array_for_prediction.astype('float32') / 255.0
            predictions = self.model.predict(img_array_for_prediction, verbose=0)[0]
            top_indices = np.argsort(predictions)[-3:][::-1]
            main_prediction = top_indices[0]
            confidence = predictions[main_prediction] * 100
            pred_name = self.character_names[main_prediction]
            # Update result labels directly (no Hindi letter)
            self.predicted_char.config(text=pred_name)
            self.char_name.config(text=pred_name)
            self.confidence.config(text=f"Confidence: {confidence:.1f}%")
            for i, idx in enumerate(top_indices):
                conf = predictions[idx] * 100
                self.top_predictions[i].config(text=f"{self.character_names[idx]} ({conf:.1f}%)")
            # Hide unused labels if less than 3
            for i in range(len(top_indices), 3):
                self.top_predictions[i].config(text="")
            # Print debug info
            print(f"Predicted: {pred_name} with {confidence:.1f}% confidence")
            print(f"Top 3: {[self.character_names[i] for i in top_indices]}")
        except Exception as e:
            print(f"Prediction error: {e}")
            messagebox.showerror("Error", f"Prediction failed: {e}")

def main():
    print("="*60)
    print("DEVANAGARI CHARACTER RECOGNITION - FIXED DRAWING GUI")
    print("="*60)
    print("Starting application...")
    
    root = tk.Tk()
    app = DevanagariDrawingApp(root)
    
    print("✓ GUI loaded successfully!")
    print("✓ Draw WHITE characters on BLACK canvas")
    print("✓ Use thick brush (10-20) for better results")
    print("✓ Try 'Toggle BG' if predictions are still wrong")
    
    root.mainloop()

if __name__ == "__main__":
    main()

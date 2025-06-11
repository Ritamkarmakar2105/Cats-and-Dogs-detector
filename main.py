import os
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
from pathlib import Path

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="RAO3qcOxTrwgNcnSGrMD"
)

# Define dataset and output paths
DATASET_PATH = r"D:\cats and dogs\dataset"
OUTPUT_PATH = r"D:\cats and dogs\output"

def load_images(dataset_path):
    """Load image paths from the dataset directory."""
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def classify_image(image_path):
    """Classify a single image and return the prediction."""
    try:
        # Perform inference
        result = CLIENT.infer(image_path, model_id="dog-and-cats/1")
        
        # Extract prediction
        if result and 'predictions' in result:
            prediction = result['predictions'][0]
            label = prediction['class']
            confidence = prediction['confidence']
            return label, confidence
        else:
            return "Unknown", 0.0
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return "Error", 0.0

def save_output_image(image_path, predicted_label, confidence, true_label, output_dir, index):
    """Save the image with prediction overlay."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    # Add prediction text
    text = f"Pred: {predicted_label} ({confidence:.2%}) | True: {true_label}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save output image
    output_filename = f"result_{index:04d}_{os.path.basename(image_path)}"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, img)
    return True

def main():
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Load all image paths
    image_paths = load_images(DATASET_PATH)
    total_images = len(image_paths)
    
    if total_images == 0:
        print("No images found in the dataset directory.")
        return
    
    print(f"Found {total_images} images to process.")
    
    # Process each image
    correct_predictions = 0
    for idx, image_path in enumerate(image_paths, 1):
        # Get the ground truth label from the filename or directory
        filename = os.path.basename(image_path).lower()
        true_label = "cat" if "cat" in filename else "dog" if "dog" in filename else "unknown"
        
        # Classify the image
        predicted_label, confidence = classify_image(image_path)
        
        # Check if prediction is correct
        is_correct = predicted_label.lower() == true_label
        if is_correct:
            correct_predictions += 1
        
        # Save the output image with prediction
        save_output_image(image_path, predicted_label, confidence, true_label, OUTPUT_PATH, idx)
        
        # Display progress
        print(f"Image {idx}/{total_images}: {os.path.basename(image_path)}")
        print(f"True Label: {true_label}")
        print(f"Predicted: {predicted_label} (Confidence: {confidence:.2%})")
        print(f"Correct: {is_correct}")
        print(f"Output saved to: {OUTPUT_PATH}\n")
    
    # Calculate and display accuracy
    accuracy = (correct_predictions / total_images) * 100
    print(f"\nClassification Complete!")
    print(f"Total Images: {total_images}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"All output images saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
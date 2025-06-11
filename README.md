# Cats-and-Dogs-detector

This project uses a pre-trained model from Roboflow to classify images of cats and dogs. It processes a dataset of images, performs inference using the Roboflow Inference API, and saves the output images with overlaid prediction results, including predicted labels, confidence scores, and ground truth labels.

Features





Loads images from a specified dataset directory.



Classifies images as "cat" or "dog" using a Roboflow model.



Overlays prediction results (predicted label, confidence) and ground truth label on each image.



Saves processed images to an output directory.



Calculates and displays classification accuracy based on ground truth labels derived from filenames.

Prerequisites





Python 3.8 or higher



A Roboflow account with an API key



A dataset of cat and dog images (images should have "cat" or "dog" in their filenames for ground truth labeling)

Installation





Clone the repository:

git clone <repository-url>
cd <repository-directory>



Install dependencies: Create a virtual environment (optional but recommended) and install the required packages:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt



Set up the dataset:





Place your dataset of cat and dog images in a directory (e.g., D:\cats and dogs\dataset).



Ensure filenames contain "cat" or "dog" to infer ground truth labels (e.g., cat_001.jpg, dog_002.png).



Configure the API key:





Obtain an API key from Roboflow.



Update the api_key in the script (RAO3qcOxTrwgNcnSGrMD) with your own key.

Usage





Update paths in the script:





Set DATASET_PATH to the directory containing your images (e.g., D:\cats and dogs\dataset).



Set OUTPUT_PATH to the directory where output images will be saved (e.g., D:\cats and dogs\output).



Run the script:

python main.py



Output:





The script processes each image, saves the results with overlaid text, and prints progress to the console.



Upon completion, it displays the total number of images, correct predictions, and accuracy.



Output images are saved in the specified OUTPUT_PATH with filenames like result_0001_image.jpg.

Example Output

Image 1/100: cat_001.jpg
True Label: cat
Predicted: cat (Confidence: 98.50%)
Correct: True
Output saved to: D:\cats and dogs\output

Classification Complete!
Total Images: 100
Correct Predictions: 95
Accuracy: 95.00%
All output images saved to: D:\cats and dogs\output

Project Structure

├── dataset/                # Directory containing input images
├── output/                 # Directory for output images with predictions
├── main.py                 # Main script for classification
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies

Dependencies

See requirements.txt for the list of required Python packages.

Notes





The ground truth label is inferred from the filename (e.g., "cat" or "dog"). Ensure your dataset follows this convention.



The Roboflow API requires an internet connection and a valid API key.



Images must be in .jpg, .jpeg, or .png format.



If an image fails to load or process, an error message is printed, and the script continues with the next image.


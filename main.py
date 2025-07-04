import os
from ultralytics import YOLO
from tkinter import filedialog, Tk
from PIL import Image
import shutil

# Load the trained YOLO model
custom_model = YOLO(r"C:\Users\ACER\runs\detect\train19\weights\best.pt")

def predict_image(image_path, model):
    """
    Predicts the class of the image using the YOLO model, saves it in the current directory,
    and opens the predicted image automatically.
    """
    try:
        # Run the model on the input image
        result = model(source=image_path, save=True, conf=0.5)
        
        # Get the directory where predictions are saved
        predict_dir = result.save_dir if hasattr(result, 'save_dir') else os.path.join("runs", "detect", "predict")
        
        # Dynamically locate the predicted image file
        predicted_image_path = None
        for file in os.listdir(predict_dir):
            if file.endswith(".jpg") or file.endswith(".png"):
                predicted_image_path = os.path.join(predict_dir, file)
                break  # Assume the first file is the one we want
        
        if predicted_image_path and os.path.exists(predicted_image_path):
            # Copy the predicted image to the current script's directory
            current_dir = os.getcwd()
            final_image_path = os.path.join(current_dir, os.path.basename(predicted_image_path))
            shutil.copy(predicted_image_path, final_image_path)

            print(f"Prediction complete. Saved at: {final_image_path}")
            
            # Automatically open the predicted image
            os.startfile(final_image_path)
        else:
            print("Prediction failed. Could not locate the predicted image. Check the YOLO output directory.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

def user_upload_and_predict():
    """
    Prompts the user to upload an image file and makes predictions using the YOLO model.
    """
    try:
        # Prompt the user to select an image
        root = Tk()
        root.withdraw()  # Hide the main Tkinter window
        file_path = filedialog.askopenfilename(
            title="Select an Image File", filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            print(f"Selected file: {file_path}")
            image = Image.open(file_path).convert('RGB')
            predict_image(file_path, custom_model)
        else:
            print("No file selected.")
    except Exception as e:
        print(f"An error occurred during file selection: {e}")

# Main execution
if __name__ == "__main__":
    print("Please select an image for prediction.")
    user_upload_and_predict()
    print("Program execution complete.")
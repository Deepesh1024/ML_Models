import pytesseract
from PIL import Image, ImageFilter
import os


def preprocess_image(image_path):
    """Preprocess the image to improve OCR accuracy.

  Args:
      image_path: Path to the image file.

  Returns:
      Preprocessed image.
  """
    try:
        # Open the image using Pillow
        img = Image.open(image_path)

        # Convert the image to grayscale
        img = img.convert('L')

        # Apply a threshold to binarize the image (black and white)
        img = img.point(lambda x: 0 if x < 128 else 255, '1')

        # Optional: Apply a filter to reduce noise
        img = img.filter(ImageFilter.MedianFilter())

        return img
    except Exception as e:
        print(f"An error occurred during image preprocessing: {e}")
        return None


def perform_ocr(image):
    """Perform OCR on a preprocessed image and return the extracted text.

  Args:
      image: Preprocessed PIL image object.

  Returns:
      Extracted text from the image.
  """
    try:
        # Perform OCR using PyTesseract
        text = pytesseract.image_to_string(image, lang='eng')
        return text
    except Exception as e:
        print(f"An error occurred during OCR: {e}")
        return "OCR failed. Please check Tesseract installation and image format."


def main(image_path):
    """Main function to perform OCR on an image.

  Args:
      image_path: Path to the image file.
  """
    # Ensure Tesseract is installed and the path is correctly set (if needed)
    # Uncomment and set the path if Tesseract is not in your system PATH
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    if preprocessed_image:
        # Perform OCR on the preprocessed image
        extracted_text = perform_ocr(preprocessed_image)
        print("Extracted Text:")
        print(extracted_text)
    else:
        print("Image preprocessing failed.")


if __name__ == "__main__":
    # Example image path (update with your actual image path)
    image_path = "/Users/deepeshjha/Desktop/SIH/ps/Screenshot 2024-08-08 at 2.20.28 PM.png"
    main(image_path)

import cv2
import pytesseract
import csv
import time

# Specify the path to the Tesseract executable if not added to the system's PATH
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Path to the Haarcascade classifier file
harcascade_path = r"/mnt/f/Detect Lisence Plate/Car-Number-Plates-Detection/model/carplate.xml"
# Create a VideoCapture object to capture from the DroidCam stream
cap = cv2.VideoCapture(2)

# Set the width and height of the captured video
cap.set(3, 640)
cap.set(4, 480)

# Minimum area for a detected plate to be considered valid
min_area = 500

# Create a CSV file to save the license plate data
output_file_path = r"/mnt/f/Detect Lisence Plate/Car-Number-Plates-Detection/license_plates.csv"

def main():
    count = 0  # Initialize count here
    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Plate Number', 'License Plate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Initialize the Haarcascade classifier
        plate_cascade = cv2.CascadeClassifier(harcascade_path)

        try:
            # Start the main loop
            while True:
                # Read a frame from the video stream
                success, img = cap.read()
                if not success:
                    break  # Exit the loop if reading the frame fails

                # Convert the frame to grayscale
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect license plates using the Haarcascade classifier
                plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

                # Process each detected plate
                for (x, y, w, h) in plates:
                    # Calculate the area of the detected plate
                    area = w * h

                    # Check if the area is above the minimum threshold
                    if area > min_area:
                        # Draw a rectangle around the detected plate
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                        # Extract the region of interest (ROI) for the plate
                        img_roi = img[y: y + h, x: x + w]

                        # Perform OCR using Tesseract
                        plate_text = pytesseract.image_to_string(img_roi)

                        # Write the recognized text to the CSV file
                        writer.writerow({'Plate Number': count + 1, 'License Plate': plate_text})

                        count += 1

                # Display the processed image
                cv2.imshow("Result", img)

                # Check for user input to exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break  # Exit the loop if 'q' or ESC is pressed

                # Add a timeout (optional)
                time.sleep(0.01)  # Wait for 0.01 seconds before checking again

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Release resources and close windows
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
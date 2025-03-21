import cv2
import time

# used  Haarcascade Face Detection Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open Webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened
if not cap.isOpened():
    print("Error: Could not access the webcam. Please check your camera.")
    exit()

print("Press 'q' to Quit | Press 's' to Save a Snapshot")

# Initialize FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame. Exiting...")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Calculate FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time

    # Display FPS on screen
    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show the output
    cv2.imshow("Face Detection - Press 'q' to Quit", frame)

    # Key Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit when 'q' is pressed
        break
    elif key == ord('s'):  # Save snapshot when 's' is pressed
        cv2.imwrite("snapshot.jpg", frame)
        print("Snapshot saved as 'snapshot.jpg'")

# Release resources
cap.release()
cv2.destroyAllWindows()

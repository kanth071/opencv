import cv2

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Open the default camera
cap = cv2.VideoCapture(0)

while True:

    # Read one frame from the camera
    ret, frame = cap.read()

    # Stop if the frame could not be read
    if not ret:
        break

    # Convert the color image to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Process every detected face
    for (x, y, w, h) in faces:

        # Draw a blue rectangle around the face
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

    # Display the result
    cv2.imshow("Face Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

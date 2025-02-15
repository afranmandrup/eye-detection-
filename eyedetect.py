import cv2

# Load the cascade
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangles around the eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the output
    cv2.imshow('img', img)
    
    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()

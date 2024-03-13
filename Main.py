import cv2

# Load pre-trained car classifier
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

# Create tracker
tracker = cv2.TrackerCSRT_create()

# Open default camera
video = cv2.VideoCapture(0)

# Read first frame
ret, frame = video.read()

# Select ROI (Region of Interest) for tracking
bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    
    # Draw rectangles around the detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Update tracker
    ok, bbox = tracker.update(frame)
    
    # Draw bounding box around tracked object
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    
    # Display the resulting frame
    cv2.imshow('Tracking', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video.release()
# Close all OpenCV windows
cv2.destroyAllWindows()

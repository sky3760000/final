# Import necessary libraries
import cv2
import numpy as np
import time
from flask import Flask, render_template, Response

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv3 model and COCO class labels
net = cv2.dnn.readNet("yolo-coco/yolov3.weights.1", "yolo-coco/yolov3.cfg")
classes = []
with open("yolo-coco/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set minimum confidence threshold and non-maximum suppression threshold
conf_threshold = 0.5
nms_threshold = 0.4

# Define function to detect objects and calculate social distance violations
def detect_objects(frame):
    # Get frame dimensions
    height, width, _ = frame.shape

    # Create blob from input frame and set as input to YOLOv3 network
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names and run forward pass through network
    output_layer_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layer_names)

    # Initialize lists to store detected objects and their properties
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each output layer and detect objects
    for output in layer_outputs:
        for detection in output:
            # Get class ID and confidence score for detected object
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections below minimum confidence threshold
            if confidence > conf_threshold:
                # Get center, width, and height of bounding box for detected object
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner of bounding box
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                # Add bounding box properties to lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Initialize list to store detected persons
    persons = []

    # Check if any objects were detected
    if len(indices) > 0:
        # Loop over each detected object and check if it is a person
        for idx in indices:
            i = idx
            if class_ids[i] == 0:
                # Get bounding box properties for person
                box = boxes[i]
                x, y, w, h = box

                # Add person to list of detected persons
                persons.append(box)

        # Initialize list to store social distance violations
        violations = []

        # Loop over each pair of detected persons and check if they violate social distance guidelines
        for i in range(len(persons)):
            for j in range(i+1, len(persons)):
                # Calculate Euclidean distance between centers of bounding boxes for two persons
                dist = np.sqrt((persons[i][0]+persons[i][2]/2 - persons[j][0]-persons[j][2]/2)**2 + (persons[i][1]+persons[i][3]/2 - persons[j][1]-persons[j][3]/2)**2)

                # Check if distance is less than minimum social distance threshold and less than 1
                if dist < 100 and dist < 1:
                    # Give alert
                    print("Social distance violation: distance is less than 1!")
                    
                # Check if distance is less than minimum social distance threshold
                elif dist < 100:
                    # Add social distance violation to list of violations
                    violations.append((i, j))

        # Draw bounding boxes and social distance violation lines on frame
        for i in range(len(boxes)):
            if i in indices:
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0) if class_ids[i] == 0 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for i, j in violations:
            x1, y1, w1, h1 = persons[i]
            x2, y2, w2, h2 = persons[j]
            cv2.line(frame, (int(x1+w1/2), int(y1+h1/2)), (int(x2+w2/2), int(y2+h2/2)), (0, 0,255), 2)

    # Return frame with bounding boxes social distance violation lines drawn
    return frame

# Define function to capture video from webcam and display on webpage
def webcam_feed():
    # Initialize video capture object
    cap = cv2.VideoCapture(0)

    # Loop over each frame from video capture object
    while True:
        # Read frame from video capture object
        ret, frame = cap.read()

        # Check if frame is valid
        if not ret:
            continue

        # Detect objects and calculate social distance violations in frame
        frame = detect_objects(frame)

        # Convert frame to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)

        # Yield frame for streaming
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        # Wait for 1 millisecond before capturing next frame
        time.sleep(0.001)

    # Release video capture object
    cap.release()

# Define route for webpage displaying video feed
@app.route('/')
def index():
    return render_template('index.html')

# Define route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(webcam_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)

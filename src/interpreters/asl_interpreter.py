import cv2 as cv
import torch
import numpy as np
from src.models.asl_model import ASLModel

def preprocess_image(frame, target_size=(244,244)):
    # Resize frame to target size
    resized_frame = cv.resize(frame, target_size)
    # Convert BGR to RGB (assuming OpenCV uses BGR by default)
    resized_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
    # Convert to tensor and normalize
    
    tensor_img = torch.tensor(resized_frame).permute(2, 0, 1).float() / 255.0
    # Add batch dimension
    tensor_img = tensor_img.unsqueeze(0)
    return tensor_img

def get_caption(output):
    # Assuming output is a tensor containing class probabilities
    _, predicted_class = torch.max(output, 1)
    
    classes = ('A', 'B','Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',)
    
    caption = classes[predicted_class.item()]
    return caption


# Load your trained model
model = ASLModel()
model.load_state_dict(torch.load('asl_model.pth'))
model.eval()

cap = cv.VideoCapture(0)  # Use 0 for the default webcam

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break

    # Preprocess the frame
    tensor_frame = preprocess_image(frame)

    # invert
    frame = cv.flip(frame, 1)

    # Pass the frame through the model
    with torch.no_grad():
        # print(model)
        prediction = model(tensor_frame)

    # Get the caption from the prediction
    caption = get_caption(prediction)

    # Display the frame with the caption
    cv.putText(frame, caption, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.imshow('ASL Interpreter', frame)

    # Press 'q' to quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv.destroyAllWindows()
import cv2 as cv
import torch
import numpy as np
from src.models.asl_model import ASLModel

def preprocess_image(frame, target_size=(244,244)):
    # Resize frame to target size
    resized_frame = cv.resize(frame, target_size)
    #  BGR to RGB (assuming OpenCV uses BGR by default)
    resized_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
    # Convert to tensor and normalize
    
    tensor_img = torch.tensor(resized_frame).permute(2, 0, 1).float() / 255.0

    tensor_img = tensor_img.unsqueeze(0)
    return tensor_img

def get_caption(output):
    _, predicted_class = torch.max(output, 1)
    
    classes = ('A', 'B','Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',)
    
    caption = classes[predicted_class.item()]
    return caption


# loading model  -- > make into main function and cleanup code everywher ewhne things get better
model = ASLModel()
model.load_state_dict(torch.load('asl_model.pth'))
model.eval()

cap = cv.VideoCapture(0)  #  0 --> defualt webacam

while True:
    ret, frame = cap.read()  # read frrame
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

    caption = get_caption(prediction)

    # dispkay capton
    cv.putText(frame, caption, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.imshow('ASL Interpreter', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

#  cleanup 
cap.release()
cv.destroyAllWindows()
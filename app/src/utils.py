
import albumentations as A # for image processing
from albumentations.pytorch import ToTensorV2
import io
import cv2
import numpy as np
# Preprocessing pipeline: Resize, Normalize for EfficientNet
def transform_image(image_bytes):
    transform = A.Compose([
        A.Resize(height=380, width=380),  # EfficientNet input size
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for pretrained models
        ToTensorV2(),
    ])
    # Convert image bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Apply transformations
    transformed = transform(image=image)
    transformed_image = transformed["image"]

    # Add batch dimension
    transformed_image = transformed_image.unsqueeze(0)

    return transformed_image
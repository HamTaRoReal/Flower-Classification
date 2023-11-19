import gradio as gr
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image

# Load Model
model = load_model('model.Flower')
# model.summary()
def classify_image(img):
  
    img = np.array(img)  # Convert PIL image to numpy array
    resize= cv2.resize(img, (224, 224))
    resize_image = np.array(resize)
    resize_image = resize_image.astype('float32') 
    resize_image /= 255     # Normalize
    resize_image = np.reshape(resize_image, (1, 224, 224, 3))
    predict = model.predict(resize_image)
    label = ['Aster', 'Daisy', 'Dandelion', 'Lavender', 'Lily', 'Marigold', 'Poppy', 'Rose', 'Sunflower', 'Tulips']
    return {label[i]: float(predict[0][i]) for i in range(len(label))}

# Create Gradio Interface
gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),  # Input image
    outputs=gr.Label(num_top_classes=10), # label
    title='Flower Classification',
    description='Input Flower Image ',
    examples=[
        ['example/Aster_0.jpeg'],['example/Aster_1.jpeg'],['example/Daisy_0.jpeg'],['example/Daisy_1.jpeg'],
        ['example/Dandelion_0.jpeg'],['example/Dandelion_1.jpeg'],['example/Lavender_0.jpeg'],['example/Lavender_1.jpeg'],
        ['example/Lily_0.jpeg'],['example/Lily_1.jpeg'],['example/Marigold_0.jpeg'],['example/Marigold_1.jpeg'],
        ['example/Poppy_0.jpeg'],['example/Poppy_1.jpeg'],['example/Rose_0.jpeg'],['example/Rose_1.jpeg'],
        ['example/Sunflower_0.jpeg'],['example/Sunflower_1.jpeg'],['example/Tulips_0.jpeg'],['example/Tulips_1.jpeg'],
        ['example/Aster_2.jpeg'],['example/Aster_3.jpeg'],['example/Daisy_2.jpeg'],['example/Daisy_3.jpeg'],
        ['example/Dandelion_2.jpeg'],['example/Dandelion_3.jpeg'],['example/Lavender_2.jpeg'],['example/Lavender_3.jpeg'],
        ['example/Lily_2.jpeg'],['example/Lily_3.jpeg'],['example/Marigold_2.jpeg'],['example/Marigold_3.jpeg'],
        ['example/Poppy_2.jpeg'],['example/Poppy_3.jpeg'],['example/Rose_2.jpeg'],['example/Rose_3.jpeg'],
        ['example/Sunflower_2.jpeg'],['example/Sunflower_3.jpeg'],['example/Tulips_2.jpeg'],['example/Tulips_3.jpeg']
    ]
).launch(share=True)


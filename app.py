# IMPORT LIBRARIES & FUNCTIONS
# External Libraries
import streamlit as st
from PIL import Image
from transfer_it import style_transfer
import cv2
import numpy as np


st.write("# Style Transfer")

st.write('''Welcome to our Style Transfer web app! With this app, you can upload your own content and style images, and generate a stylized output image using the neural style transfer algorithm. 
         Simply upload your images, click the 'Transfer Style' button, and wait for the output image to be generated. You can also clear your inputs by clicking the 'Clear Inputs' button.''')


content_path_default = "demo/content.jpeg"
style_path_default = "demo/style.jpeg"

col1, col2 = st.columns(2)

# Add file pickers for content and style images
content_path = col1.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_path = col2.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

# Check if user has uploaded new images or using defaults
if content_path is None:
    content_path = content_path_default
if style_path is None:
    style_path = style_path_default

content_image = Image.open(content_path)
col1.image(content_image, caption="Content Image", use_column_width=True)
style_image = Image.open(style_path)
col2.image(style_image, caption="Style Image", use_column_width=True)

# Add button to generate stylized output image
output_shown = False


if st.button("Transfer Style"):
    with st.spinner("Transferring style"):
        output_path = "output/output.jpg"
        style_transfer(content_path, style_path)
        output_image = Image.open(output_path)
        st.image(output_image, caption="Original", width=256)
        
        output_shown = True

# Show "Clear Inputs" button if output is being displayed
if output_shown:
    if st.button("Clear outputs"):
        output_image = None
        output_shown = False

st.write("Inference might be slow as the model is running on a cpu instance.")

# IMPORT LIBRARIES & FUNCTIONS
# External Libraries
import streamlit as st
from PIL import Image
from transfer_it import style_transfer, device_info
import cv2
import numpy as np


st.write("# Style Transfer")

st.write('''Welcome to our Style Transfer web app! With this app, you can upload your own content and style images, and generate a stylized output image using a transformer model. 
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

def apply_filters(img_path, filter_name,alpha=0.5):
    if isinstance(img_path,str):
        img = cv2.imread(img_path)
    else:
        img = Image.open(img_path)
        img = img.save("output/up_content.jpg")
        img = cv2.imread("output/up_content.jpg")
    
    if filter_name == "sobel":
        # Apply Sobel filter
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        filtered_img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        output_path = "output/pre/c_sobel.jpeg"

    elif filter_name == "gaussian":
        # Apply Gaussian filter
        filtered_img = cv2.GaussianBlur(img, (5, 5), 0)
        output_path = "output/pre/c_gaussian.jpeg"

    elif filter_name == "median":
        filtered_img = cv2.medianBlur(img, 5)
        output_path = "output/pre/c_median.jpeg" 

    elif filter_name == "bilateral":
        filtered_img = cv2.bilateralFilter(img, 9, 75, 75)
        output_path = "output/pre/c_bilateral.jpeg"

    elif filter_name == "equalize":
        # Apply histogram equalization
        b, g, r = cv2.split(img)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        filtered_img = cv2.merge([b_eq, g_eq, r_eq])
        output_path = "output/pre/c_equalized.jpeg"

    elif filter_name == "edgedetection":
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply the Canny edge detection filter
        edges = cv2.Canny(gray, 100, 200)
        # Create a mask for the edges
        mask = np.zeros_like(img)
        mask[edges != 0] = 255
        # Apply the mask to the original image
        filtered_img = cv2.bitwise_and(img, mask)
        output_path = "output/pre/c_edgedetection.jpeg"

    elif filter_name == "sharpen":
        # Define the kernel for the sharpen filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        filtered_img = cv2.filter2D(img, -1, kernel)
        output_path = "output/pre/c_sharpen.jpeg"
    else:
        raise ValueError("Invalid filter name. Valid options are 'sobel', 'gaussian', and 'equalize'.")

    # Superimpose the filtered image on the original image
    output_img = cv2.addWeighted(img, alpha, filtered_img, 1 - alpha, 0)

    # Save the filtered image
    cv2.imwrite(output_path, output_img)

    return output_path


if st.button("Transfer Style"):
    with st.spinner("Transferring style"):
        output_path = "output/output.jpg"

        col1, col2, col3= st.columns(3)
        style_transfer(content_path, style_path)
        output_image = Image.open(output_path)
        col1.image(output_image, caption="Original", width=256)

        sobel_path = apply_filters(content_path,'sobel')
        style_transfer(sobel_path, style_path)
        output_image = Image.open(output_path)
        col2.image(output_image, caption="/w Sobel Filter", width=256)

        gaussian_path= apply_filters(content_path,'gaussian')
        style_transfer(gaussian_path, style_path)
        output_image = Image.open(output_path)
        col3.image(output_image, caption="/w Gaussian Filter", width=256)

        equalized_path = apply_filters(content_path,'equalize')
        style_transfer(equalized_path, style_path)
        output_image = Image.open(output_path)
        col1.image(output_image, caption="/w Histogram Equalization", width=256)
        
        median_path = apply_filters(content_path,'median')
        style_transfer(median_path, style_path)
        output_image = Image.open(output_path)
        col2.image(output_image, caption="/w Median Filter", width=256)

        bilateral_path = apply_filters(content_path,'bilateral')
        style_transfer(bilateral_path, style_path)
        output_image = Image.open(output_path)
        col3.image(output_image, caption="/w Bilateral Filter", width=256)

        edgedetection_path = apply_filters(content_path,'edgedetection')
        style_transfer(edgedetection_path, style_path)
        output_image = Image.open(output_path)
        col1.image(output_image, caption="/w Edge Detection", width=256)

        sharpen_path = apply_filters(content_path,'sharpen')
        style_transfer(sharpen_path, style_path)
        output_image = Image.open(output_path)
        col2.image(output_image, caption="/w Sharpen Filter", width=256)
        print('Done')  
        output_shown = True

# Show "Clear Inputs" button if output is being displayed
if output_shown:
    if st.button("Clear outputs"):
        output_image = None
        output_shown = False

device = device_info()

if str(device) == "cpu":
    st.write("Inference might be slow as the model is running on a cpu instance.")

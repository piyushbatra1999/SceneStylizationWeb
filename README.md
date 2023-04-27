# Scene Stylization

This project implements an image style transfer application using a transformer model. Users can upload their content and style images, and generate a stylized output image using the implemented model. The project is written in Python 3, using PyTorch, and Streamlit libraries.

### Project Structure

- demo/: contains sample content and style images for demonstration purposes
- networks/: contains pre-trained models for the image style transfer task
- transfer_it.py: style transfer inference utilities
- app.py: implementation of the Streamlit web application
- function.py: implementation of auxiliary functions used by the style transfer model
- models/: network implementation files
- output/: directory for storing preprocessed and output images
- util/: directory for storing network utility functions

### Prerequisites

- Python 3
- PyTorch
- Streamlit
- Pillow
- Matplotlib
- NumPy
- OpenCV

### Usage

```python
streamlit run app.py
```

This will launch a local server and open the web application in your default browser.

To use the application, follow these steps:

- Upload a content image and a style image using the respective file pickers.
- Click the "Transfer Style" button to generate a stylized output image.
- Wait for the output images to be generated.
- The output images will be displayed on the web page.
- You can also clear your inputs by clicking the "Clear Inputs" button.

### Acknowledgments

The style transfer model implemented in this project is based on the works of Y. Deng et al., “StyTr$^2$: Image Style Transfer with Transformers,” arXiv.org, 2021. https://arxiv.org/abs/2105.14576.
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

# Function to convert an image with alpha channel
def convert_to_rgba(image):
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        return image.convert('RGBA')
    else:
        return image.convert('RGB')

# Function to apply border detection
def apply_border_detection(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# Main Streamlit app
st.title('Image Processing App')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Check image format
    file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type,
                    "filesize": uploaded_file.size}
    st.write(file_details)

    # Load and display image
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    st.image(image, caption='Uploaded Image', use_column_width=False, width=300)

    # Create columns for tweaks and results
    col1, col2 = st.columns(2)

    with col1:
        # Opacity slider
        opacity = st.slider("Adjust opacity", 0.0, 1.0, 1.0)
        if image.mode != 'RGBA':
            image = image.convert('RGBA')  # Ensure the image is in RGBA format
        
        alpha = image.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
        image.putalpha(alpha)

        # Color adjustment sliders
        st.header("Adjust Colors")
        red = st.slider("Red", 0, 255, 100)
        green = st.slider("Green", 0, 255, 100)
        blue = st.slider("Blue", 0, 255, 100)
        img_array = np.array(image)
        img_array[:,:,0] = img_array[:,:,0] * (red / 100.0)
        img_array[:,:,1] = img_array[:,:,1] * (green / 100.0)
        img_array[:,:,2] = img_array[:,:,2] * (blue / 100.0)
        image = Image.fromarray(np.uint8(img_array))

    with col2:
        st.image(image, caption='Adjusted Image', use_column_width=False, width=300)

    # Download final image
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Adjusted Image",
        data=byte_im,
        file_name="adjusted_image.png",
        mime="image/png",
    )

    if st.button('Add Border Detection'):
        edges = apply_border_detection(image_array)
        
        st.image(edges, caption='Edges Detected Image', use_column_width=False,  width=300,clamp=True, channels='GRAY')
        
        buf_edges = io.BytesIO()
        edges_image = Image.fromarray(edges)
        edges_image.save(buf_edges, format="PNG")
        byte_edges_im = buf_edges.getvalue()

        st.download_button(
            label="Download Edges Image",
            data=byte_edges_im,
            file_name="edges_detected_image.png",
            mime="image/png",
        )

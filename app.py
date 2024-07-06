import streamlit as st
from PIL import Image
from utils.utils import stylize, image_preprocess, resize_image_proportionally

# Initialize session state for file_name
if "file_name" not in st.session_state:
    st.session_state["file_name"] = "output_image"


content_image = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
style_options = ["The Scream", "The Starry Night", "The Dance", "The Papal Palace", "Louis Valtat",
                 "Diana Malivani", "Kangchenjunga"]
style_choice = st.selectbox("Choose a style", style_options)

model_paths = {
    "The Scream": "models/epoch_500_1_100000.0_style1.model",
    "The Starry Night": "models/epoch_500_1_100000.0_style2.model",
    "The Dance": "models/epoch_200_1_100000.0_style3.model",
    "The Papal Palace": "models/epoch_200_1_100000.0_style4.model",
    "Louis Valtat": "models/epoch_200_1_100000.0_style5.model",
    "Diana Malivani": "models/epoch_200_1_100000.0_style6.model",
    "Kangchenjunga": "models/epoch_200_1_100000.0_style7.model"
}

style_photo_paths = {
    "The Scream": "images/style_images/style1.jpg",
    "The Starry Night": "images/style_images/style2.jpg",
    "The Dance": "images/style_images/style3.jpg",
    "The Papal Palace": "images/style_images/style4.jpg",
    "Louis Valtat": "images/style_images/style5.jpg",
    "Diana Malivani": "images/style_images/style6.jpg",
    "Kangchenjunga": "images/style_images/style7.jpg"
}

if content_image and style_choice:
    model_path = model_paths[style_choice]
    style_photo_path = style_photo_paths[style_choice]

    style_image = Image.open(style_photo_path)
    style_image = resize_image_proportionally(style_image, 512)

    # Get the file name from session state
    st.session_state["file_name"] = st.text_input("Enter the file name", st.session_state["file_name"])

    if st.button("Apply Style"):
        with st.spinner('Applying style...'):
            output_image = stylize(content_image, model_path)
            output_image = image_preprocess(output_image)

            col1, col2 = st.columns(2)
            col1.image(content_image, caption="Original image")
            col2.image(output_image, caption="Styled image")

            st.image(style_image, caption="Style image", use_column_width=True)

            output_image_path = f"images/downloads/{st.session_state['file_name']}.png"
            output_image.save(output_image_path)

            with open(output_image_path, "rb") as file:
                st.download_button("Download styled image", file, file_name=f"{st.session_state['file_name']}.png", mime="image/png")

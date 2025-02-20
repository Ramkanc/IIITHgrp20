import streamlit as st
from app_6_1 import cos_8k_image_to_text, cos_8k_text_to_image, cos_8k_image_to_image_top
from app_6_2 import cos_30k_image_to_text, cos_30k_text_to_image, cos_30k_image_to_image_top
from app_6_3 import generate_caption
from app_6_4 import generate_transform_caption

st.markdown(
    """
    <style>
    .big-font {
        font-size:30px !important;
        font-family: 'Helvetica', Courier, monospace;
        font-weight: bold;
        color: #7b0214;
    }
    .caption {
        font-size:20px !important;
        font-family: 'Helvetica', Courier, monospace;
        font-weight: bold;
        color: #7b0214;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define the functions for each option
def cosine_similarity_8k_image_to_text(image):
    #st.write("Processing image to text using CosineSimilarity8k...")
    # Add your image-to-text processing code here
    caption, idx = cos_8k_image_to_text(image)
    #st.write(f"Most similar caption: {caption}")
    st.markdown(f'<p class="caption">Most similar caption: {caption} </p>', unsafe_allow_html=True)
    st.write(f"Index: {idx}")
def cosine_similarity_8k_text_to_image(caption):
    st.write("Processing text to image using CosineSimilarity8k...")
    # Add your text-to-image processing code here
    most_similar_image_path,idx = cos_8k_text_to_image(caption)
    st.image(most_similar_image_path, caption="Most similar image", use_container_width=True)
    st.write(f"Most similar index: {idx}")

def cosine_similarity_30k_image_to_text(image):
    st.write("Processing image to text using CosineSimilarity30k...")
    # Add your image-to-text processing code here
    caption, idx = cos_30k_image_to_text(image)
    #st.write(f"Most similar caption: {caption}")
    st.markdown(f'<p class="caption">Most similar caption: {caption} </p>', unsafe_allow_html=True)
    st.write(f"Index: {idx}")

def cosine_similarity_30k_text_to_image(caption):
    st.write("Processing text to image using CosineSimilarity30k...")
    # Add your text-to-image processing code here
    most_similar_image_path,idx = cos_30k_text_to_image(caption)
    st.image(most_similar_image_path, caption="Most similar image", use_container_width=True)    
    st.write(f"Most similar index: {idx}")

def clip_vitb32_lstm_attention_image_to_text(image):
    st.write("Processing image to text using CLIPViTB32->LSTM:Attention...")
    # Add your image-to-text processing code here
    caption = generate_caption(image)
    #st.write(f"Generated Caption: {caption}")
    st.markdown(f'<p class="caption">Generated Caption: {caption} </p>', unsafe_allow_html=True)

def clip_vitb32_lstm_attention_text_to_image(caption):
    st.write("Processing text to image using CLIPViTB32->LSTM:Attention...")
    # Add your text-to-image processing code here

def clip_vitb32_gpt2_image_to_text(image):
    st.write("Processing image to text using CLIPViTB32->GPT2...")
    # Add your image-to-text processing code here
    caption = generate_transform_caption(image)
    #st.write(f"Generated Caption: {caption}")
    st.markdown(f'<p class="caption">Generated Caption: {caption} </p>', unsafe_allow_html=True)

def clip_vitb32_gpt2_text_to_image(caption):
    st.write("Processing text to image using CLIPViTB32->GPT2...")
    # Add your text-to-image processing code here

def cosine_similarity_8k_image_to_image(image):
    st.write("Processing iamge to Image using CosineSimilarity8k...")
    # Add your image-to-image processing code here
    most_similar_image_path, idx = cos_8k_image_to_image_top(image)
    st.image(most_similar_image_path, caption="Most similar image", use_container_width=True)
    st.write(f"Most similar index: {idx}")

def cosine_similarity_30k_image_to_image(image):
    st.write("Processing iamge to Image using CosineSimilarity30k...")
    # Add your image-to-image processing code here
    most_similar_image_path, idx = cos_30k_image_to_image_top(image)
    st.image(most_similar_image_path, caption="Most similar image", use_container_width=True)
    st.write(f"Most similar index: {idx}")

def clip_vitb32_lstm_attention_image_to_image(image):
    pass

def clip_vitb32_gpt2_image_to_image(image):
    pass



# Streamlit UI
st.markdown('<p class="big-font">Image Captioning App: Group 20</p>', unsafe_allow_html=True)

# Sidebar with dropdown list
st.sidebar.title("Options")
dropdown_options = ["CosineSimilarity8k", "CosineSimilarity30k", "CLIPViTB32->LSTM:Attention", "CLIPViTB32->GPT2"]
selected_option = st.sidebar.selectbox("Select an option", dropdown_options)
st.sidebar.title("Image and Text Processing")
task = st.sidebar.radio("Choose a task", ("Image-to-Text", "Text-to-Image", "Image-to-Image"))

# Upload image
if task in ["Image-to-Text" , "Image-to-Image"]:
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        image = uploaded_image
    else:
        captured_image = st.camera_input("Capture an image")
        if captured_image is not None:
            st.image(captured_image, caption="Captured Image", use_container_width=True)
            image = captured_image
        else:
            image = None 


# Enter caption
if task == "Text-to-Image":
    caption = st.text_input("Enter a caption")
    if caption:
        st.write(f"Entered Caption: {caption}")

# Placeholder for further processing
if st.button("Process"):
    if task == "Image-to-Text" and image is not None:
        if selected_option == "CosineSimilarity8k":
            cosine_similarity_8k_image_to_text(image)
        elif selected_option == "CosineSimilarity30k":
            cosine_similarity_30k_image_to_text(image)
        elif selected_option == "CLIPViTB32->LSTM:Attention":
            clip_vitb32_lstm_attention_image_to_text(image)
        elif selected_option == "CLIPViTB32->GPT2":
            clip_vitb32_gpt2_image_to_text(image)
    elif task == "Text-to-Image" and caption:
        if selected_option == "CosineSimilarity8k":
            cosine_similarity_8k_text_to_image(caption)
        elif selected_option == "CosineSimilarity30k":
            cosine_similarity_30k_text_to_image(caption)
        elif selected_option == "CLIPViTB32->LSTM:Attention":
            clip_vitb32_lstm_attention_text_to_image(caption)
        elif selected_option == "CLIPViTB32->GPT2":
            clip_vitb32_gpt2_text_to_image(caption)
    elif task == 'Image-to-Image' and image is not None:
        if selected_option == "CosineSimilarity8k":
            cosine_similarity_8k_image_to_image(image)
        elif selected_option == "CosineSimilarity30k":
            cosine_similarity_30k_image_to_image(image)
        elif selected_option == "CLIPViTB32->LSTM:Attention":
            clip_vitb32_lstm_attention_image_to_image(image)
        elif selected_option == "CLIPViTB32->GPT2":
            clip_vitb32_gpt2_image_to_image(image)

    else:
        st.write("Please upload an image or enter a caption to proceed.")
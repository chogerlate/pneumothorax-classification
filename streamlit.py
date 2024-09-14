import streamlit as st
import requests
import os

def classify_images(files):
    """Send images to FastAPI service and return predictions."""
    try:
        # Construct the full FastAPI URL for the /predict endpoint
        API_URL = os.path.join(os.getenv("API_URL", "http://127.0.0.1:8000"), "predict")
        
        # Prepare the file objects for sending in the POST request
        file_data = [("files", (file.name, file, file.type)) for file in files]

        # Make POST request to the FastAPI service with the files
        response = requests.post(API_URL, files=file_data, timeout=60)  # 60s timeout for large files
        response.raise_for_status()  # Raise an error for bad responses

        # Extract and return the predictions from the response JSON
        return response.json().get("predictions", [])

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the server: {str(e)}")
        return None




def display_predictions(predictions, uploaded_files):
    """Display predictions with a clear layout."""
    st.markdown("### Classification Results")

    for idx, result in enumerate(predictions):
        st.markdown("---")  # Add a separator between predictions

        col1, col2 = st.columns([1, 2])

        with col1:
            # Display the image from uploaded files
            st.image(uploaded_files[idx], caption=result['filename'], use_column_width=True)

        with col2:
            pneumothorax_prob = result['pneumothorax_prob']
            diagnosis = result['diagnosis']

            # Display prediction info
            st.markdown(f"**Filename:** `{result['filename']}`")
            st.markdown(f"**Pneumothorax Probability:** `{pneumothorax_prob:.2f}`")

            # Display progress bar for probability
            st.progress(pneumothorax_prob)

            # Display diagnosis result with color-coded messages
            if diagnosis == 'Pneumothorax':
                st.error("⚠️ **Positive for Pneumothorax**")
            else:
                st.success("✅ **No Pneumothorax Detected**")

    st.markdown("---")  # Final separator

def main():
    # Set the page layout
    st.set_page_config(page_title="Pneumothorax Classification Service", page_icon="🩺", layout="wide")

    # Title and introduction
    st.title("🩺 Pneumothorax Classification Service")
    st.markdown("""
    **Upload your chest X-ray images** below to classify whether the patient has Pneumothorax. 
    This tool helps healthcare professionals analyze chest X-rays quickly.
    """)

    # Sidebar: Instructions and file uploader
    with st.sidebar:
        st.header("Upload Images")
        st.markdown("""
        **How to use:**
        - Upload multiple chest X-ray images.
        - Click 'Classify Images' to see the predictions.
        """)
        uploaded_files = st.file_uploader(
            "Choose images (jpg, jpeg, png)", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )

    # Main section: Display uploaded images or a prompt to upload
    if uploaded_files:
        st.markdown("### Uploaded Images")
        cols = st.columns(4)  # Organize images in grid format

        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % 4]:
                st.image(
                    uploaded_file,
                    caption=f"Image: {uploaded_file.name}",
                    use_column_width=True
                )

        # Button to classify images
        if st.button("🚀 Classify Images"):
            with st.spinner("Classifying..."):
                predictions = classify_images(uploaded_files)
                if predictions:
                    display_predictions(predictions, uploaded_files)
                else:
                    st.error("No predictions returned from the server.")
    else:
        st.markdown("### No images uploaded yet.")
        st.info("Please upload images from the sidebar to start the classification.")

if __name__ == "__main__":
    main()

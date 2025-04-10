import streamlit as st
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model import PrescriptionModel, NLPTextExtractor, ReminderGenerator
from paddleocr import PaddleOCR
import json
from datetime import datetime
import uuid
import numpy as np
import nltk
import cv2
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Load spaCy model for NLP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Set page config
st.set_page_config(
    page_title="Prescription Processing System",
    page_icon="💊",
    layout="wide"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'reminders' not in st.session_state:
    st.session_state.reminders = None

# Initialize components
@st.cache_resource
def load_model():
    try:
        model = PrescriptionModel()
        if os.path.exists('best_prescription_model.pth'):
            model.load_state_dict(torch.load('best_prescription_model.pth'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_ocr():
    try:
        # Initialize OCR with absolute minimum settings
        ocr = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            show_log=False,
            use_gpu=False,
            enable_mkldnn=False,
            cpu_threads=1,
            det_algorithm='DB',  # Use basic detection algorithm
            rec_algorithm='CRNN',  # Use basic recognition algorithm
            det_limit_side_len=960,  # Limit image size
            det_limit_type='min',
            rec_batch_num=1  # Process one image at a time
        )
        return ocr
    except Exception as e:
        st.error(f"Error initializing OCR: {str(e)}")
        return None

@st.cache_resource
def load_nlp():
    try:
        return NLPTextExtractor()
    except Exception as e:
        st.error(f"Error loading NLP: {str(e)}")
        return None

# Load components
model = load_model()
ocr = load_ocr()
nlp = load_nlp()

if not all([model, ocr, nlp]):
    st.error("Failed to initialize one or more components. Please check the error messages above.")
    st.stop()

def process_image(image):
    """Process prescription image and extract information"""
    try:
        # Convert PIL Image to numpy array for OCR
        image_np = np.array(image)
        
        # Basic image preprocessing
        if len(image_np.shape) == 3:
            # Convert to grayscale
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Resize image if too large
        max_size = 960
        height, width = image_np.shape
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            image_np = cv2.resize(image_np, None, fx=scale, fy=scale)
        
        # Basic image enhancement
        image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX)
        
        # Simple thresholding
        _, image_np = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Perform OCR with error handling
        try:
            # Convert back to PIL Image for OCR
            image_pil = Image.fromarray(image_np)
            
            # Perform OCR with minimal settings
            result = ocr.ocr(image_pil, cls=False)
            
            if not result:
                return None, "OCR failed to process the image. Please try again with a clearer image."
            if not result[0]:
                return None, "No text was detected in the image. Please check if the prescription is clearly visible."
                
            # Extract text with validation
            try:
                extracted_text = "\n".join([line[1][0] for line in result[0] if isinstance(line, (list, tuple)) and len(line) > 1])
                if not extracted_text.strip():
                    return None, "No text content was extracted. Please ensure the prescription text is visible."
            except Exception as text_error:
                st.error(f"Text extraction error: {str(text_error)}")
                return None, "Error extracting text from the image."
                
            # Process with NLP
            try:
                medicine_data = nlp.extract_medicine_data(extracted_text)
                if not medicine_data:
                    return None, "No medicine information could be extracted. Please ensure the prescription contains clear medicine details."
            except Exception as nlp_error:
                st.error(f"NLP processing error: {str(nlp_error)}")
                return None, "Error processing the extracted text."
                
            return medicine_data, extracted_text
            
        except Exception as ocr_error:
            st.error(f"OCR processing error: {str(ocr_error)}")
            return None, "Error during OCR processing. Please try again with a clearer image."
            
    except Exception as e:
        st.error(f"General processing error: {str(e)}")
        return None, "An unexpected error occurred. Please try again with a different image."

def generate_reminders(medicine_data, user_id):
    """Generate reminders for the extracted medicine data"""
    try:
        reminder_generator = ReminderGenerator(user_id=user_id)
        reminders = reminder_generator.schedule_reminders(medicine_data)
        return reminders
    except Exception as e:
        st.error(f"Error generating reminders: {str(e)}")
        return None

# Main UI
st.title("💊 Prescription Processing System")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Process Prescription", "View Reminders", "About"])

if page == "Process Prescription":
    st.header("Upload Prescription")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a prescription image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Prescription", use_column_width=True)
        
        # Process button
        if st.button("Process Prescription"):
            with st.spinner("Processing prescription..."):
                # Process image
                medicine_data, extracted_text = process_image(image)
                
                if medicine_data:
                    st.session_state.processed_data = medicine_data
                    
                    # Display extracted text
                    with col2:
                        st.subheader("Extracted Text")
                        st.text_area("OCR Result", extracted_text, height=200)
                    
                    # Display medicine information
                    st.subheader("Extracted Medicine Information")
                    for idx, medicine in enumerate(medicine_data, 1):
                        with st.expander(f"Medicine {idx}: {medicine['name']}"):
                            st.write(f"**Dosage:** {medicine.get('dosage', 'N/A')}")
                            st.write(f"**Frequency:** {medicine.get('frequency', 'N/A')}")
                            st.write(f"**Duration:** {medicine.get('duration', 'N/A')}")
                    
                    # Reminder generation
                    st.subheader("Schedule Reminders")
                    user_id = st.text_input("Enter User ID", value=str(uuid.uuid4()))
                    
                    if st.button("Generate Reminders"):
                        with st.spinner("Generating reminders..."):
                            reminders = generate_reminders(medicine_data, user_id)
                            if reminders:
                                st.session_state.reminders = reminders
                                st.success(f"Successfully generated {len(reminders)} reminders!")
                else:
                    st.error("Failed to process prescription. Please try again with a clearer image.")

elif page == "View Reminders":
    st.header("Scheduled Reminders")
    
    if st.session_state.reminders:
        for reminder in st.session_state.reminders:
            with st.expander(f"Reminder for {reminder['medicine']}"):
                st.write(f"**Dosage:** {reminder['dosage']}")
                st.write(f"**Frequency:** {reminder['frequency']}")
                st.write(f"**Duration:** {reminder['duration']}")
                st.write(f"**Next Reminder:** {reminder['reminder_time'].strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Status:** {reminder['status']}")
    else:
        st.info("No reminders have been generated yet. Process a prescription first.")

else:  # About page
    st.header("About")
    st.write("""
    ### Prescription Processing System
    
    This application helps you process medical prescriptions and manage medication reminders.
    
    **Features:**
    - OCR-based prescription text extraction
    - NLP-powered medicine information extraction
    - Automated reminder generation
    - User-friendly interface
    
    **How to use:**
    1. Upload a prescription image
    2. Click 'Process Prescription' to extract information
    3. Review the extracted data
    4. Generate reminders if needed
    
    **Technologies used:**
    - PyTorch for deep learning
    - PaddleOCR for text extraction
    - NLTK and spaCy for NLP
    - Streamlit for the web interface
    """)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ for better healthcare management") 
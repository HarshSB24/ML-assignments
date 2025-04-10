# VAMD - Visual Analysis of Medical Documents

VAMD is a prescription processing system that uses computer vision, OCR, NLP, and deep learning to extract medicine information from prescription images and generate reminders.

## Features

- **OCR Processing**: Extract text from prescription images using PaddleOCR
- **NLP-based Medicine Extraction**: Identify medicine names, dosages, frequencies, and durations using NLP techniques
- **Deep Learning Model**: Classify prescription images to identify medicine types using a PyTorch model
- **Reminder Generation**: Schedule reminders based on extracted medicine data
- **User Preferences**: Customize reminder times and other settings
- **Web Interface**: Flask-based web application for easy interaction

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/VAMD.git
cd VAMD
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

4. Download spaCy model:
```
python -m spacy download en_core_web_sm
```

## Usage

### Running the Web Application

```
python app.py
```

The web application will be available at http://localhost:5000

### Training the Model

To train the PyTorch model on your own dataset:

1. Prepare your dataset with prescription images
2. Run the model.py script and select option 2 for training
3. Enter the path to your dataset directory
4. The trained model will be saved as 'best_prescription_model.pth'

### Testing the Model

To test the model on a prescription image:

```
python test_model.py
```

Follow the prompts to enter the path to a test prescription image.

## Project Structure

- `app.py`: Flask web application
- `model.py`: PyTorch model and NLP text extractor
- `test_model.py`: Script to test the model on prescription images
- `templates/`: HTML templates for the web interface
- `static/`: CSS, JavaScript, and other static files
- `uploads/`: Directory for uploaded prescription images
- `reminders/`: Directory for generated voice reminders

## Model Architecture

The PyTorch model uses a pre-trained ResNet50 architecture with the following modifications:

- Modified final layer for multi-label classification
- Sigmoid activation for multi-label output
- Dropout for regularization

## NLP Text Extraction

The NLP text extractor uses the following techniques:

- spaCy for text processing and entity recognition
- Section-based extraction to identify different medicines
- Pattern matching for dosage, frequency, and duration extraction

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PaddleOCR for OCR processing
- spaCy for NLP processing
- PyTorch for deep learning
- Flask for web framework 
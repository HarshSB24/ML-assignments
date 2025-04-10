import os
import json
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from paddleocr import PaddleOCR
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.metrics import f1_score
import logging
import matplotlib.pyplot as plt
import gc
from functools import lru_cache
import uuid
import spacy
from gtts import gTTS
from apscheduler.schedulers.background import BackgroundScheduler

# Load spaCy model for NLP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)8s] %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('prescription_process.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class PrescriptionDataset(Dataset):
    def __init__(self, image_dir, cache_dir, transform=None, force_reload=False, 
                 batch_process=True, batch_size=50, image_size=(400, 600)):
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self.transform = transform
        self.force_reload = force_reload
        self.batch_process = batch_process
        self.batch_size = batch_size
        self.image_size = image_size
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Process images and cache results
        if force_reload or not self._check_cache_exists():
            logger.info("Processing images and caching results...")
            self._preprocess_images()
            
        # Load cached results
        self.cached_results = self._load_cached_results()
        
        # Print dataset statistics
        self.print_statistics()

    def _check_cache_exists(self):
        # Check if a sample of files exists to speed up initialization
        sample_size = min(100, len(self.image_files))
        sample_files = self.image_files[:sample_size]
        
        return all(
            os.path.exists(os.path.join(self.cache_dir, f"{os.path.splitext(img)[0]}.json"))
            for img in sample_files
        )

    def _preprocess_images(self):
        successful = 0
        failed = 0
        
        # Initialize OCR only once
        ocr = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            show_log=False,
            use_gpu=torch.cuda.is_available(),
            enable_mkldnn=True,
            cpu_threads=4
        )
        
        # Process images in batches
        if self.batch_process:
            remaining_files = list(self.image_files)
            
            while remaining_files:
                # Take a batch of files
                batch_files = remaining_files[:self.batch_size]
                remaining_files = remaining_files[self.batch_size:]
                
                for img_file in tqdm(batch_files, desc=f"Processing batch ({len(batch_files)} images)"):
                    try:
                        # Skip if already cached
                        cache_path = os.path.join(self.cache_dir, f"{os.path.splitext(img_file)[0]}.json")
                        if os.path.exists(cache_path):
                            successful += 1
                            continue
                            
                        # Process image
                        image_path = os.path.join(self.image_dir, img_file)
                        result = self._process_single_image(image_path, ocr)
                        
                        # Cache result
                        with open(cache_path, 'w') as f:
                            json.dump(result, f)
                        
                        successful += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing {img_file}: {str(e)}")
                        failed += 1
                        continue
                
                # Force garbage collection between batches
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Process images one by one (original approach)
            for img_file in tqdm(self.image_files, desc="Processing images"):
                try:
                    # Skip if already cached
                    cache_path = os.path.join(self.cache_dir, f"{os.path.splitext(img_file)[0]}.json")
                    if os.path.exists(cache_path):
                        successful += 1
                        continue
                        
                    # Process image
                    image_path = os.path.join(self.image_dir, img_file)
                    result = self._process_single_image(image_path, ocr)
                    
                    # Cache result
                    with open(cache_path, 'w') as f:
                        json.dump(result, f)
                    
                    successful += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {str(e)}")
                    failed += 1
                    continue
        
        logger.info(f"\nProcessing complete:")
        logger.info(f"Successfully processed: {successful}")
        logger.info(f"Failed to process: {failed}")
        logger.info(f"Success rate: {(successful/(successful+failed))*100:.2f}%")

    def _process_single_image(self, image_path, ocr):
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return {'text': [], 'label': [0] * 7}
                
            # Use a smaller resize to reduce memory usage
            image = cv2.resize(image, self.image_size)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply simpler preprocessing to save memory
            # Skip adaptive histogram equalization (CLAHE) which is memory-intensive
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Use simple thresholding instead of adaptive thresholding
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Perform OCR with optimized parameters
            try:
                result = ocr.ocr(binary, cls=False)
                # Force garbage collection after OCR to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"OCR failed for {image_path}: {str(e)}")
                return {'text': [], 'label': [0] * 7}
            
            # Handle None or empty results
            if result is None or len(result) == 0 or not result[0]:
                logger.warning(f"No OCR results for {image_path}")
                return {'text': [], 'label': [0] * 7}
                
            # Extract text and confidence with validation
            texts = []
            confidences = []
            try:
                for line in result[0]:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                        confidence = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.0
                        texts.append(text)
                        confidences.append(confidence)
            except Exception as e:
                logger.warning(f"Error extracting text from OCR result for {image_path}: {str(e)}")
                return {'text': [], 'label': [0] * 7}
            
            # Generate labels based on OCR text
            label = self._generate_labels(texts)
            
            return {
                'text': texts,
                'confidence': confidences,
                'label': label
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return {'text': [], 'label': [0] * 7}

    @lru_cache(maxsize=128)
    def _normalize_text(self, text):
        """Normalize text for better matching, with caching for performance"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove common prefixes and suffixes
        prefixes = ['tab', 'tablet','TAB','Tab','Cap','CAP', 'cap', 'capsule', 'syrup', 'SYR','Syrup','Syr','suspension', 'INJ','Inj','inj', 'injection']
        for prefix in prefixes:
            if text.startswith(prefix + ' '):
                text = text[len(prefix):].strip()
        
        # Remove special characters and extra spaces
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        text = ' '.join(text.split())
        
        return text

    def _generate_labels(self, texts):
        # Initialize label vector
        label = [0] * 7
        
        # Common medication form indicators with variations
        forms = {
            0: {  # Tablets
                'keywords': ['tablet','Tablet', 'Tab','TAB','tab','PILL','Pill', 'pill'],
                'variations': ['tablets', 'tabs', 'pills']
            },
            1: {  # Capsules
                'keywords': ['capsule','Capsule','CAP','Cap', 'cap'],
                'variations': ['capsules', 'caps']
            },
            2: {  # Syrups
                'keywords': ['syrup','Syrup','SYR','Syr', 'suspension'],
                'variations': ['syrups', 'suspensions']
            },
            3: {  # Injections
                'keywords': ['injection','Injection','INJECTION','INJ', 'inj'],
                'variations': ['injections', 'injs']
            },
            4: {  # Drops
                'keywords': ['drops', 'eye drops'],
                'variations': ['drop', 'eyedrops']
            },
            5: {  # Topical
                'keywords': ['cream', 'ointment'],
                'variations': ['creams', 'ointments']
            },
            6: {  # Inhalers/Sprays
                'keywords': ['inhaler', 'spray'],
                'variations': ['inhalers', 'sprays']
            }
        }
        
        # Process each text line
        for text in texts:
            # Normalize text
            normalized_text = self._normalize_text(text)
            
            # Check for medication forms
            for form_id, form_info in forms.items():
                # Check main keywords
                if any(keyword in normalized_text for keyword in form_info['keywords']):
                    label[form_id] = 1
                    break
                
                # Check variations
                if any(variation in normalized_text for variation in form_info['variations']):
                    label[form_id] = 1
                    break
                
                # Check for exact matches (case-insensitive)
                if any(keyword.lower() in text.lower() for keyword in form_info['keywords']):
                    label[form_id] = 1
                    break
                
                # Check for variations with exact matches
                if any(variation.lower() in text.lower() for variation in form_info['variations']):
                    label[form_id] = 1
                    break
        
        return label

    def _load_cached_results(self):
        cached_results = {}
        for img_file in self.image_files:
            cache_path = os.path.join(self.cache_dir, f"{os.path.splitext(img_file)[0]}.json")
            try:
                if os.path.exists(cache_path):
                    with open(cache_path, 'r') as f:
                        cached_results[img_file] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache for {img_file}: {str(e)}")
                continue
        return cached_results

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_file)
        
        # Load image with PIL and handle memory issue
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a black image in case of error
            image = torch.zeros(3, 224, 224)
        
        # Get cached result
        result = self.cached_results.get(img_file, {'label': [0] * 7})
        
        return {
            'image': image,
            'label': torch.tensor(result['label'], dtype=torch.float32)
        }

    def print_statistics(self):
        """Print dataset statistics with proper error handling"""
        logger.info("\nDataset Statistics:")
        logger.info(f"Total images: {len(self.image_files)}")
        logger.info(f"Cached results: {len(self.cached_results)}")
        
        # Print sample label distribution
        labels = [result['label'] for result in self.cached_results.values()]
        if not labels:
            logger.warning("No labels found in the dataset!")
            return
            
        labels = np.array(labels)
        if labels.size == 0:
            logger.warning("Empty labels array!")
            return
            
        if len(labels.shape) < 2:
            logger.warning(f"Invalid labels shape: {labels.shape}")
            return
            
        logger.info("\nLabel Distribution:")
        for i in range(labels.shape[1]):
            positive_count = np.sum(labels[:, i] == 1)
            logger.info(f"Class {i}: {positive_count} positive samples ({positive_count/len(labels)*100:.2f}%)")

class PrescriptionModel(nn.Module):
    """
    PyTorch model for prescription medicine classification
    """
    def __init__(self, num_classes=7):
        super(PrescriptionModel, self).__init__()
        
        # Use a pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the final layer for our classification task
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # Use sigmoid for multi-label classification
        )
    
    def forward(self, x):
        return self.resnet(x)

class PrescriptionTrainer:
    """
    Trainer class for the prescription model
    """
    def __init__(self, model, train_loader, val_loader, device, learning_rate=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Define loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        f1 = f1_score(all_labels, np.array(all_preds) > 0.5, average='weighted')
        
        return avg_loss, f1
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        f1 = f1_score(all_labels, np.array(all_preds) > 0.5, average='weighted')
        
        return avg_loss, f1
    
    def train(self, num_epochs=10, patience=5):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_loss, train_f1 = self.train_epoch()
            
            # Validation phase
            val_loss, val_f1 = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_prescription_model.pth')
                logger.info("Model saved!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

class NLPTextExtractor:
    """
    Class for extracting medicine information from prescription text using NLP techniques
    """
    def __init__(self):
        self.medicine_types = {
            'tablet': ['tab', 'tablet', 'pill'],
            'capsule': ['cap', 'capsule'],
            'syrup': ['syrup', 'suspension', 'syr'],
            'injection': ['inj', 'injection'],
            'drops': ['drops', 'eye drops'],
            'cream': ['cream', 'ointment'],
            'inhaler': ['inhaler', 'spray']
        }
        
        # Common dosage units
        self.dosage_units = ['mg', 'g', 'ml', 'mcg', 'tablets', 'tablet', 'capsules', 'capsule']
        
        # Common frequency patterns
        self.frequency_patterns = [
            'once a day', 'twice a day', 'three times a day', 'four times a day',
            '1x', '2x', '3x', '4x', '1 time', '2 times', '3 times', '4 times'
        ]
        
        # Common duration patterns
        self.duration_patterns = [
            'days', 'weeks', 'months', 'day', 'week', 'month'
        ]
    
    def extract_medicine_data(self, text):
        """
        Extract medicine information from text using spaCy NLP
        """
        medicine_data = []
        
        # Split text into lines
        lines = text.split('\n')
        
        # Process each line to identify medicine sections
        current_medicine = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new medicine section
            is_new_medicine = False
            for med_types in self.medicine_types.values():
                for med_type in med_types:
                    if line.lower().startswith(med_type.lower()):
                        is_new_medicine = True
                        break
                if is_new_medicine:
                    break
            
            if is_new_medicine:
                # If we have a current medicine, add it to the list
                if current_medicine and current_medicine["name"]:
                    medicine_data.append(current_medicine)
                
                # Start a new medicine entry
                current_medicine = {
                    "name": "",
                    "dosage": "",
                    "frequency": "",
                    "duration": ""
                }
                
                # Extract medicine name from this line
                parts = line.split()
                for i, part in enumerate(parts):
                    if any(med_type in part.lower() for med_types in self.medicine_types.values() for med_type in med_types):
                        if i + 1 < len(parts):
                            # Join remaining parts as the medicine name
                            name_parts = parts[i+1:]
                            current_medicine["name"] = ' '.join(name_parts).replace(' - ', '-').strip()
                            break
            
            # If we have a current medicine, try to extract other information
            if current_medicine:
                # Check for dosage
                if "tablets" in line.lower() or "tablet" in line.lower() or "ml" in line.lower():
                    # Extract dosage
                    doc = nlp(line)
                    for token in doc:
                        if token.like_num:
                            try:
                                quantity = int(token.text)
                                # Look for the unit
                                if token.i + 1 < len(doc):
                                    unit = doc[token.i+1].text.lower()
                                    if "tablet" in unit or "ml" in unit:
                                        current_medicine["dosage"] = f"{quantity} {unit}"
                                        break
                            except ValueError:
                                continue
                
                # Check for frequency
                if "times" in line.lower() or "once" in line.lower() or "twice" in line.lower():
                    # Extract frequency
                    doc = nlp(line)
                    for token in doc:
                        if token.like_num:
                            try:
                                times = int(token.text)
                                if token.i + 1 < len(doc) and "times" in doc[token.i+1].text.lower():
                                    current_medicine["frequency"] = f"{times} times a day"
                                    break
                            except ValueError:
                                continue
                    
                    # Check for text-based frequency
                    if not current_medicine["frequency"]:
                        if "once" in line.lower():
                            current_medicine["frequency"] = "once a day"
                        elif "twice" in line.lower():
                            current_medicine["frequency"] = "twice a day"
                        elif "three times" in line.lower():
                            current_medicine["frequency"] = "three times a day"
                
                # Check for duration
                if "days" in line.lower() or "day" in line.lower():
                    # Extract duration
                    doc = nlp(line)
                    for token in doc:
                        if token.like_num:
                            try:
                                days = int(token.text)
                                if token.i + 1 < len(doc) and "day" in doc[token.i+1].text.lower():
                                    current_medicine["duration"] = f"{days} days"
                                    break
                            except ValueError:
                                continue
        
        # Add the last medicine if it exists
        if current_medicine and current_medicine["name"]:
            medicine_data.append(current_medicine)
        
        return medicine_data

class ReminderGenerator:
    """
    Class for generating and scheduling reminders based on extracted medicine data
    """
    def __init__(self, user_id, prescription_id=None, config=None):
        self.user_id = user_id
        self.prescription_id = prescription_id
        self.reminders_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reminders")
        os.makedirs(self.reminders_dir, exist_ok=True)
        
        # Default configuration
        self.config = {
            'default_duration': 7,  # Default duration in days
            'default_method': 'After Food',  # Default method
            'default_status': 'upcoming',  # Default status
            'reminder_prefix': 'It\'s time to take',  # Prefix for reminder text
            'reminder_suffix': '',  # Suffix for reminder text
            'audio_format': 'mp3',  # Audio format for voice reminders
        }
        
        # Update with user-provided configuration
        if config:
            self.config.update(config)
    
    def generate_voice_reminder(self, text, reminder_id):
        """Generate voice reminder for the given text."""
        try:
            output_file = os.path.join(self.reminders_dir, f"reminder_{reminder_id}.{self.config['audio_format']}")
            tts = gTTS(text=text, lang='en')
            tts.save(output_file)
            return f"reminder_{reminder_id}.{self.config['audio_format']}"
        except Exception as e:
            print(f"Error generating audio: {e}")
            return None
    
    def schedule_reminders(self, medicine_data):
        """
        Schedule reminders based on extracted medicine data
        """
        scheduled_reminders = []
        
        for medicine in medicine_data:
            # Extract medicine information
            name = medicine.get("name", "").strip()
            dosage = medicine.get("dosage", "").strip()
            frequency = medicine.get("frequency", "").lower().strip()
            duration = medicine.get("duration", "").lower().strip()
            method = medicine.get("method", self.config['default_method'])
            
            # Determine times per day
            times_per_day = 1
            if "twice" in frequency or "2x" in frequency or "2 times" in frequency:
                times_per_day = 2
            elif "three" in frequency or "3x" in frequency or "3 times" in frequency:
                times_per_day = 3
            elif "four" in frequency or "4x" in frequency or "4 times" in frequency:
                times_per_day = 4
            
            # Set default reminder times based on times per day
            reminder_times = []
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            if times_per_day == 1:
                reminder_times.append(today.replace(hour=9, minute=0))  # 9 AM
            elif times_per_day == 2:
                reminder_times.append(today.replace(hour=9, minute=0))  # 9 AM
                reminder_times.append(today.replace(hour=21, minute=0))  # 9 PM
            elif times_per_day == 3:
                reminder_times.append(today.replace(hour=9, minute=0))  # 9 AM
                reminder_times.append(today.replace(hour=14, minute=0))  # 2 PM
                reminder_times.append(today.replace(hour=21, minute=0))  # 9 PM
            elif times_per_day == 4:
                reminder_times.append(today.replace(hour=8, minute=0))  # 8 AM
                reminder_times.append(today.replace(hour=12, minute=0))  # 12 PM
                reminder_times.append(today.replace(hour=16, minute=0))  # 4 PM
                reminder_times.append(today.replace(hour=20, minute=0))  # 8 PM
            
            # Determine duration in days
            duration_days = self.config['default_duration']  # Default duration
            if "week" in duration:
                try:
                    weeks = int(duration.split()[0])
                    duration_days = weeks * 7
                except:
                    pass
            elif "day" in duration:
                try:
                    duration_days = int(duration.split()[0])
                except:
                    pass
            
            # Set start and end times
            start_time = datetime.now()
            end_time = start_time + timedelta(days=duration_days)
            
            # Schedule reminders
            for day in range(duration_days):
                for base_time in reminder_times:
                    reminder_time = base_time + timedelta(days=day)
                    if reminder_time > datetime.now() and reminder_time < end_time:
                        reminder_id = str(uuid.uuid4())
                        reminder_text = f"{self.config['reminder_prefix']} {dosage} of {name}"
                        if self.config['reminder_suffix']:
                            reminder_text += f" {self.config['reminder_suffix']}"
                        
                        # Generate voice reminder
                        audio_file = self.generate_voice_reminder(reminder_text, reminder_id)
                        
                        # Create reminder object
                        reminder = {
                            'id': reminder_id,
                            'user_id': self.user_id,
                            'medicine': name,
                            'dosage': dosage,
                            'frequency': frequency,
                            'duration': duration,
                            'method': method,
                            'reminder_time': reminder_time,
                            'status': self.config['default_status'],
                            'audio_file': audio_file,
                            'prescription_id': self.prescription_id
                        }
                        
                        # Schedule job
                        scheduler.add_job(
                            func=lambda r=reminder: self.trigger_reminder(r['id']),
                            trigger="date",
                            run_date=reminder_time,
                            id=reminder_id
                        )
                        
                        scheduled_reminders.append(reminder)
        
        return scheduled_reminders
    
    def trigger_reminder(self, reminder_id):
        """
        Trigger reminder when time is hit
        """
        print(f"REMINDER ALERT: Reminder {reminder_id} triggered")
        return reminder_id

def main():
    try:
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # GPU setup and memory management
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            torch.cuda.empty_cache()
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU")
        
        # Initialize OCR
        logger.info("Initializing OCR...")
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        # Initialize NLP components
        logger.info("Initializing NLP components...")
        text_extractor = NLPTextExtractor()
        
        while True:
            print("\n=== Prescription Processing System ===")
            print("1. Process prescription image")
            print("2. Exit")
            
            choice = input("\nEnter your choice (1-2): ")
            
            if choice == "2":
                print("Exiting...")
                break
                
            elif choice == "1":
                # Get prescription image path from user
                image_path = input("\nEnter the path to your prescription image: ")
                
                if not os.path.exists(image_path):
                    print("Error: File does not exist!")
                    continue
                
                try:
                    # Process the image with OCR
                    logger.info(f"Processing image: {image_path}")
                    result = ocr.ocr(image_path, cls=True)
                    
                    if not result or not result[0]:
                        print("Error: Could not extract text from the image!")
                        continue
                    
                    # Extract text from OCR result
                    extracted_text = "\n".join([line[1][0] for line in result[0]])
                    print("\nExtracted Text:")
                    print("---------------")
                    print(extracted_text)
                    
                    # Extract medicine data using NLP
                    logger.info("Extracting medicine data...")
                    extracted_medicine_data = text_extractor.extract_medicine_data(extracted_text)
                    
                    if not extracted_medicine_data:
                        print("\nNo medicine information found in the prescription!")
                        continue
                    
                    print("\nExtracted Medicine Data:")
                    print("----------------------")
                    for idx, medicine in enumerate(extracted_medicine_data, 1):
                        print(f"\nMedicine {idx}:")
                        print(f"Name: {medicine.get('name', 'N/A')}")
                        print(f"Dosage: {medicine.get('dosage', 'N/A')}")
                        print(f"Frequency: {medicine.get('frequency', 'N/A')}")
                        print(f"Duration: {medicine.get('duration', 'N/A')}")
                    
                    # Ask user if they want to schedule reminders
                    schedule_choice = input("\nWould you like to schedule reminders for these medicines? (y/n): ")
                    
                    if schedule_choice.lower() == 'y':
                        # Get user ID (in a real application, this would come from authentication)
                        user_id = input("Enter your user ID: ")
                        
                        # Initialize reminder generator
                        logger.info("Initializing reminder generator...")
                        reminder_generator = ReminderGenerator(user_id=user_id)
                        
                        # Schedule reminders
                        logger.info("Scheduling reminders...")
                        scheduled_reminders = reminder_generator.schedule_reminders(extracted_medicine_data)
                        
                        print(f"\nSuccessfully scheduled {len(scheduled_reminders)} reminders!")
                        print("\nReminder Schedule:")
                        print("-----------------")
                        for reminder in scheduled_reminders:
                            print(f"\nMedicine: {reminder['medicine']}")
                            print(f"Dosage: {reminder['dosage']}")
                            print(f"Time: {reminder['reminder_time'].strftime('%Y-%m-%d %H:%M')}")
                            print(f"Status: {reminder['status']}")
                    
                except Exception as e:
                    logger.error(f"Error processing prescription: {str(e)}")
                    print(f"Error: {str(e)}")
            
            else:
                print("Invalid choice! Please try again.")
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
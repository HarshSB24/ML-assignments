import random
from PIL import Image, ImageDraw, ImageFont
import os
import json

# Create folders for prescriptions and labels if they don't exist
if not os.path.exists('test_prescriptions'):
    os.makedirs('test_prescriptions')

if not os.path.exists('test_labels'):
    os.makedirs('test_labels')

# Function to generate random details for prescriptions
def generate_random_details():
    # Define random sets for patient details, medicine, etc.
    first_names = ["John", "Jane", "Alice", "Bob", "Charlie", "David", "Eve", "Grace", "Liam", "Mia"]
    last_names = ["Doe", "Roe", "Smith", "Brown", "White", "Johnson", "Williams", "Miller", "Davis", "Wilson"]
    medicines = [
        "Tab Ciplox 500", "Tab Combiflam", "Tab Pan d", "Tab Althrocin 500", "Tab Febrex plus", 
        "Tab Sumo cold", "Tab Dolonex dt", "Tab Clavum 625", "Tab Dolo 650", "Tab Triz", 
        "Tab Cetzine", "Tab Allegra", "Tab Zole f cream", "Tab Cyclopam", "Tab Cyclomeff", 
        "Tab Evion", "Tab Neurobin Forte", "Cap Omez", "Cap Co-Immune", "Cap Zovirax", 
        "Cap Dilantin", "Cap Pan d", "Cap Cypon"
    ]
    syrups = [
        "Syrup TusQ dx", "Syrup Cypon", "Syrup Grilinctus bm", "Syrup Febrex plus", "Syrup Corex", 
        "Syrup Corex-dx", "Syrup Zedex", "Syrup TusQ-dx", "Syrup TusQ-x", "Syrup Brozedex"
    ]
    dosages = ["5 tablets", "10 tablets", "15 tablets", "20 tablets"]
    syrup_dosages = ["5ml", "10ml"]
    frequencies = ["2 times a day", "3 times a day"]
    durations = ["5 days", "7 days", "10 days", "14 days", "30 days"]
    doctors = ["Dr. Smith", "Dr. Johnson", "Dr. Patel", "Dr. Lee", "Dr. Miller", "Dr. Turner"]

    # Randomly choose details from the lists
    patient_name = f"{random.choice(first_names)} {random.choice(last_names)}"
    age = random.randint(20, 80)

    # Generate medicines with separate dosages for tablets/capsules and syrups
    medicines_list = random.sample(medicines, random.randint(4, 6))
    syrups_list = random.sample(syrups, random.randint(1, 2))
    dosages_list = [random.choice(dosages) for _ in medicines_list]
    syrup_dosages_list = [random.choice(syrup_dosages) for _ in syrups_list]
    frequencies_list = [random.choice(frequencies) for _ in medicines_list]
    syrup_frequencies_list = [random.choice(frequencies) for _ in syrups_list]
    durations_list = [random.choice(durations) for _ in medicines_list]
    syrup_durations_list = [random.choice(durations) for _ in syrups_list]
    doctor = random.choice(doctors)

    return patient_name, age, medicines_list, syrups_list, dosages_list, syrup_dosages_list, frequencies_list, syrup_frequencies_list, durations_list, syrup_durations_list, doctor

# Function to create prescription PNG and label
def create_prescription_and_label(patient_name, age, medicines, syrups, dosages, syrup_dosages, frequencies, syrup_frequencies, durations, syrup_durations, doctor, filename):
    # Create an image
    img = Image.new('RGB', (800, 1200), color='white')
    draw = ImageDraw.Draw(img)

    # Load default font
    header_font = ImageFont.truetype("arial.ttf", 24)
    section_font = ImageFont.truetype("arial.ttf", 20)
    normal_font = ImageFont.truetype("arial.ttf", 16)

    # Add header
    draw.rectangle([(0, 0), (800, 100)], fill="lightblue")
    draw.text((50, 20), "XYZ Hospital", font=header_font, fill="black")
    draw.text((50, 50), "123 Health Street, Wellness City", font=normal_font, fill="black")
    draw.text((50, 70), "Phone: +1-234-567-890", font=normal_font, fill="black")

    # Add patient and doctor details
    draw.text((50, 120), f"Patient Name: {patient_name}", font=normal_font, fill="black")
    draw.text((50, 150), f"Age: {age}", font=normal_font, fill="black")
    draw.text((50, 180), f"Doctor: {doctor}", font=normal_font, fill="black")

    # Medicine table header
    y = 230
    draw.rectangle([(50, y), (750, y + 30)], fill="lightgrey")
    draw.text((55, y + 5), "Medicine", font=normal_font, fill="black")
    draw.text((250, y + 5), "Dosage", font=normal_font, fill="black")
    draw.text((350, y + 5), "Frequency", font=normal_font, fill="black")
    draw.text((500, y + 5), "Duration", font=normal_font, fill="black")

    # Draw medicines table
    y += 30
    medicine_labels = []
    for medicine, dosage, frequency, duration in zip(medicines, dosages, frequencies, durations):
        draw.rectangle([(50, y), (750, y + 30)], outline="black", width=1)
        draw.text((55, y + 5), medicine, font=normal_font, fill="black")
        draw.text((250, y + 5), dosage, font=normal_font, fill="black")
        draw.text((350, y + 5), frequency, font=normal_font, fill="black")
        draw.text((500, y + 5), duration, font=normal_font, fill="black")
        y += 30
        # Save label for each medicine
        medicine_labels.append({
            "name": medicine,
            "dosage": dosage,
            "frequency": frequency,
            "duration": duration
        })

    # Syrup table header
    y += 20
    draw.rectangle([(50, y), (750, y + 30)], fill="lightgrey")
    draw.text((55, y + 5), "Syrup", font=normal_font, fill="black")
    draw.text((250, y + 5), "Dosage", font=normal_font, fill="black")
    draw.text((350, y + 5), "Frequency", font=normal_font, fill="black")
    draw.text((500, y + 5), "Duration", font=normal_font, fill="black")

    # Draw syrups table
    y += 30
    syrup_labels = []
    for syrup, dosage, frequency, duration in zip(syrups, syrup_dosages, syrup_frequencies, syrup_durations):
        draw.rectangle([(50, y), (750, y + 30)], outline="black", width=1)
        draw.text((55, y + 5), syrup, font=normal_font, fill="black")
        draw.text((250, y + 5), dosage, font=normal_font, fill="black")
        draw.text((350, y + 5), frequency, font=normal_font, fill="black")
        draw.text((500, y + 5), duration, font=normal_font, fill="black")
        y += 30
        # Save label for each syrup
        syrup_labels.append({
            "name": syrup,
            "dosage": dosage,
            "frequency": frequency,
            "duration": duration
        })
    # Add footer
    y += 40
    draw.rectangle([(0, y), (800, y + 100)], fill="lightgrey")
    draw.text((50, y + 10), "Advice:", font=section_font, fill="black")
    draw.text((70, y + 40), "- Take bed rest", font=normal_font, fill="black")
    draw.text((70, y + 60), "- Eat easy-to-digest food like boiled rice with dal", font=normal_font, fill="black")
    draw.text((70, y + 80), "- Stay hydrated", font=normal_font, fill="black")

    # Save prescription image
    img.save(f'test_prescriptions/{filename}.png')

    # Save labels in JSON file
    label_data = {
        "medicines": medicine_labels,
        "syrups": syrup_labels
    }

    with open(f'test_labels/{filename}.json', 'w') as label_file:
        json.dump(label_data, label_file, indent=4)

# Function to generate random prescriptions and labels
def generate_random_prescriptions(num_samples):
    for i in range(1, num_samples + 1):
        # Generate random details for each prescription
        patient_name, age, medicines, syrups, dosages, syrup_dosages, frequencies, syrup_frequencies, durations, syrup_durations, doctor = generate_random_details()
        
        # Filename based on sequential numbering
        filename = f"{i}"
        
        # Create and save prescription PNG and label
        create_prescription_and_label(patient_name, age, medicines, syrups, dosages, syrup_dosages, frequencies, syrup_frequencies, durations, syrup_durations, doctor, filename)

# Ask user for the number of prescriptions to generate
if __name__ == "__main__":
    num_images = int(input("Enter the number of prescription PNGs to generate: "))
    generate_random_prescriptions(num_images)
    print(f"{num_images} prescriptions generated successfully in the 'test_prescriptions' folder and labels in the 'test_labels' folder!")

function handleFileUpload(event) {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/predict', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        const extractedData = data.extracted_medicine_data;
        const medicineList = document.getElementById('medicineList');
        
        extractedData.forEach(medicine => {
            const listItem = document.createElement('li');
            listItem.textContent = medicine;
            medicineList.appendChild(listItem);
        });
    })
    .catch(error => console.error('Error:', error));
}

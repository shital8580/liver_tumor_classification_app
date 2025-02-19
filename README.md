🧬 Liver Tumor Classification Web Application
This is a web application designed to classify liver tumors using histopathological images. The application uses a pre-trained DenseNet121 deep learning model and is built with Streamlit.

🚀 Features
✅ Upload histopathological images of the liver.
✅ Predict tumor type using the trained DenseNet121 model.
✅ User-friendly web interface powered by Streamlit.

🛠️ Requirements
Python 3.8+
Streamlit
PyTorch
Torchvision

⚙️ Installation
Clone the repository:git clone https://github.com/shital8580/liver_tumor_classification_app.git

Navigate to the project directory:cd liver_tumor_classification_app

Install the dependencies:pip install -r requirements.txt

🖥️ Running the Application
To run the Streamlit app:streamlit run app.py

📖 Usage
Open the web app in your browser.
Upload a histopathological image.
View the predicted liver tumor class.

🧠 Model Details
Architecture: DenseNet121
Preprocessing: Image resizing, normalization, and augmentation applied.
Training: Conducted on labeled histopathological images with multi-class classification.
Loss Function: Cross-entropy loss
Optimizer: Adam optimizer with learning rate scheduling

🗄️ Dataset
The dataset contains histopathological images of liver tissues categorized into multiple tumor classes.
Images underwent preprocessing and augmentation to improve model generalization.

📊 Results
Accuracy: Achieved high accuracy on the validation set.

Deployment: Successfully deployed via Streamlit with a responsive user interface.


Image Classification: Age and Gender Detection
This project implements an image classification system to predict both age and gender from facial images. The model uses a pre-trained VGG16 network for feature extraction, with custom layers for age and gender classification.


The system processes facial images and predicts age categories (Kid, Teenager, Young, Adult, Senior, Elderly) and gender (Male/Female). It is divided into three main stages:

Preprocessing: Images are resized, normalized, and categorized.
Training: A neural network is trained using a combination of VGG16's pre-trained layers and custom layers for age and gender classification.
Testing and Evaluation: The trained model is tested on unseen images, and its predictions are evaluated.
Installation
To get started with this project:

Clone the repository:

git clone https://github.com/Izdibayev23/Image-Classification.git
cd image-classification
Create a virtual environment and install dependencies:

python3 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
Make sure you have the necessary image datasets organized as per the naming convention (age_gender.jpg).

Usage
Run the provided Jupyter notebooks for different tasks:

Preprocessing: Preprocess the dataset for training.
jupyter notebook Preprocessing.ipynb

Training: Train the classification model.
jupyter notebook Training.ipynb

Testing: Evaluate the model on test data.
jupyter notebook Testing.ipynb

Real-time Webcam Feed: Use your webcam to capture images and view results.
jupyter notebook Real-time.ipynb

Project Structure

Preprocessing.ipynb: Preprocesses the images and splits the dataset.
Training.ipynb: Implements model training using VGG16 as a base.
Testing.ipynb: Tests the model on unseen test images.
Real-time.ipynb: Captures images from a webcam in real-time and saves screenshots.
Webcam.ipynb: Displays the webcam feed without saving images.

Dataset
The dataset is expected to be organized as follows:

bash
Копировать код
/train
  age_gender.jpg (e.g., 25_1.jpg where 25 is the age, and 1 represents Male)
/val
/test
Age: Numerical value representing the age.
Gender: Binary label (0 for Male, 1 for Female).
Results
After training, the model outputs two predictions per image:

Age: Categorized as one of the six age groups (Kid, Teenager, Young, Adult, Senior, Elderly).
Gender: Predicted as either Male or Female.
You can also run the webcam interface to capture live video feed and predict age and gender in real-time.

Future Improvements
Implement data augmentation to improve model robustness.
Fine-tune the model to handle imbalanced datasets more effectively.
Explore other pre-trained architectures or self-supervised learning techniques.
Add more fine-grained age categories or gender representations.
Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request to improve the project.

# wasteSort
Introduction:
This project aims to create a waste segregation deep learning model using Convolutional Neural Networks (CNN). The model classifies waste items into six categories: Cardboard, Glass, Metal, Paper, Plastic, and Trash.

Project Overview:
Mount Google Drive: The code mounts Google Drive to access the dataset and save the trained model.
Data Visualization: It visualizes sample images from the dataset for each waste category using matplotlib.
Data Preprocessing: The dataset is prepared using the ImageDataGenerator class from Keras, which preprocesses and augments the image data.
Model Architecture: The CNN model architecture consists of convolutional layers followed by max-pooling layers for feature extraction, and dense layers for classification.
Model Training: The model is trained using the ImageDataGenerator flow_from_directory method, which generates batches of augmented data for training.
Model Evaluation: The trained model's performance is evaluated using the test dataset to measure accuracy and loss.
Model Prediction: Test images are loaded, preprocessed, and fed into the trained model for prediction. Predicted classes and probabilities are displayed for each test image.
Model Saving: After training, the model is saved to the specified location for future use.
Code Structure:
Imports: Import necessary libraries including TensorFlow, Keras, matplotlib, and PIL.
Google Drive Mounting: Mount Google Drive to access dataset and save model.
Data Visualization: Visualize sample images from the dataset.
Data Preparation: Preprocess and augment image data using ImageDataGenerator.
Model Definition: Define the CNN model architecture.
Model Compilation: Compile the model with optimizer, loss function, and metrics.
Model Training: Train the model using the prepared data.
Model Evaluation: Evaluate the model's performance on the test dataset.
Model Prediction: Load test images, preprocess, predict classes, and visualize predictions.
Model Saving: Save the trained model to a specified location.
Usage:
Ensure access to Google Drive containing the dataset.
Adjust file paths if necessary.
Run the code cells sequentially to mount Google Drive, preprocess data, define and train the model, and save the model.
Evaluate model performance and make predictions on test images.
Conclusion:
The waste segregation deep learning model successfully classifies waste items into six categories, promoting efficient waste management and recycling efforts.

This this a project to complete the competition published on Kaggle: 
Plant Seedlings Classification - Determine the species of a seedling from an image

The aim of this competition is to differentiate crop seedlings from 12 plant categories using datasets provided by Aarhus University
The difficulty of this competition is that the original img are in different sizes with angel of rotations and flips. Thus, hyperparameters tuning needs to be done carefully to have a high accuracy predication result.

The original page for the description of the competition can be found here:
https://www.kaggle.com/competitions/plant-seedlings-classification/

Part A: brief description of the organization of the project
This project retrained a deep learning CNN model by using tensorflow.keras and one of its original CNN network - InceptionResNetV2.
InceptionResNetV2 is a 164 layers deep network and it's suitable to deal with classification problem with many categories
The model could classify 12 categories of plant seedings provided in advance with high accuracy and save the prediction result for test datasets in a csv file.

Part B: the role of each file
In plant-seedings-classification folder: it contains 4750 images of plant seedlings in 12 categories for training and 794 images for testing
main.py includes the model design, model training and validation
prediction.py includes the testing process and saves the prediction result in a csv file
test_result.csv contains the testing result
requirements.txt contains all the packages needed to be installed

Part C: the packages required
tensorflow, keras, pandas, matplotlib

run the following line to install all the packages needed
pip install -r requirements.txt



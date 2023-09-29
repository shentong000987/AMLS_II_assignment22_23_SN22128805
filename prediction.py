import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def predict():
    # desired img size after resizing
    img_width = 200
    img_height = 200
    seedlings_name = [
        'Black-grass', 
        'Charlock', 
        'Cleavers', 
        'Common Chickweed', 
        'Common wheat', 
        'Fat Hen', 
        'Loose Silky-bent', 
        'Maize', 
        'Scentless Mayweed', 
        'Shepherds Purse', 
        'Small-flowered Cranesbill', 
        'Sugar beet'
    ]

    test_result_file = 'test_result.csv'

    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_directory(
        directory = 'plant-seedlings-classification',
        classes=['test'],
        target_size = (img_width, img_height),
        batch_size = 1,
        class_mode = "categorical",
        shuffle = False
    )

    model =  tf.keras.models.load_model('plant_seedlings_classification.keras')
    predictions = model.predict(test_generator, steps=test_generator.samples)

    # input the testing result to a csv file
    class_list = []

    for i in range(0, predictions.shape[0]):
        y_class = predictions[i, :].argmax(axis=-1)
        class_list += [seedlings_name[y_class]]

    testing_result = pd.DataFrame()
    testing_result['filename'] = test_generator.filenames
    testing_result['filename'] = testing_result['filename'].str.replace(r'test/', '')
    testing_result['classification'] = class_list

    testing_result.to_csv(test_result_file, index=False)




if __name__ == "__main__":
    predict()
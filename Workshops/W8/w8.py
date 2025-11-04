from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model
import pandas as pd
import numpy as np 
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
#Import the logging library
import logging

#Heuristic Applied: Configure logging to include timestamp, level, and message
# This will assist in establish a timeline for forensic analysis
logging.basicConfig(level=logging.INFO, filename='ml_security.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

def readData():
    # Location 1: Logging the start of a data loading operation.
    # Justification: Tracks the initation of a critical process (data ingestion).
    logging.info("readData - Initiating data loading from sklearn.datasets.")

    iris = datasets.load_iris()
    
    # Location 2: Logging the source and type of the loaded dataset.
    # Justification: Records data provenance. Will let you know which dataset is used during a poisoning attack.
    logging.info(f"readData - Successfully loaded 'iris' dataset. Type: {type(iris)}")

    print(type(iris.data), type(iris.target))
    X = iris.data
    Y = iris.target
    
    # Location 3: Logging the shape of the features and target arrays.
    # Justification: A sudden, unexpected change in data dimensions can indicate a data tampering or poisoning attack.
    logging.info(f"readData - Dataset dimensions: Features shap {X.shape}, Target shape {Y.shape}.")

    df = pd.DataFrame(X, columns=iris.feature_names)
    print(df.head())

    # Location 4: Logging the successful creation of the DataFrame.
    # Justification: Confirms that the raw data was successfully parsed into a structured format, a critical step in the data pipeline.
    logging.info("readData - Successfully created pandas DataFrame from iris data.")

    return df 

def makePrediction():
    # Location 5: Logging the start of the prediction function.
    # Justification: Marks the beginning of a prediction task, helping the isolate events related to this specific operation.
    logging.info("makePrediction - Initiating prediction task.")
    
    iris = datasets.load_iris()

    # Location 6: Logging the model's hyperparameters.
    # Justification: Records the configuration of the model. Unauthorized changes to hyperparameters can be a form of attack.
    logging.info("makePrediction - Initializing KNeighborsClassifier with n_neighbors=6.")
    knn = KNeighborsClassifier(n_neighbors=6)

    # Location 7: Logging the start of the model fitting (training) process.
    # Justification: Pinpoints the moment training begins. If the process fails or behavior is abnormal, this log is a starting point.
    logging.info("makePrediction - Starting model fitting on iris data.")
    knn.fit(iris['data'], iris['target'])

    # Location 8: Logging the successful completion of model fitting.
    # Justification: Confirms the model has been trained, a prerequisite for making predictions.
    logging.info("makePrediction - Model fitting completed successfully.")

    X = [
        [5.9, 1.0, 5.1, 1.8],
        [3.4, 2.0, 1.1, 4.8],
    ]

    # Location 9: Logging the input data sent for prediction.
    # Justification: Crucial for detecting model tricking. This log records the exact input that may have caused an erroneous prediction.
    logging.info(f"makePrediction - Predicting for input data: {X}")
    prediction = knn.predict(X)

    # Location 10: Logging the output of a prediction.
    # Justification: Records the model's decision. This can be compared against expected outcomes to detect model tricking.
    logging.info(f"makePrediction - Prediction successful. Result: {prediction}")
    print(prediction)    

def doRegression():
    # Location 11: Logging the start of the regression task.
    # Justification: Provides context that a regression model is now being trained and used.
    logging.info("doRegression - Initiating regression task.")
    diabetes = datasets.load_diabetes()

    diabetes_X = diabetes.data[:, np.newaxis, 2]
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # Location 12: Logging the size of training and test datasets.
    # Justification: Verifies the integrity of the data split. An unusual split could indicate an attempt to influence the model's performance.
    logging.info(f"doRegression - Data split into training ({len(diabetes_X_train)}) and testing ({len(diabetes_X_test)}) sets.")

    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Location 13: Logging a key model attribute after training.
    # Justification: Model coefficients are the result of training. A drastic, unexplained change in coefficients over time could signal, a data poisoning attack.
    logging.info(f"doRegression - Model fit complete. Regression coefficient: {regr.coef_}")

    diabetes_y_pred = regr.predict(diabetes_X_test)

    # Location 14: Logging the successful completion of the regression prediction.
    # Justification: Confirms that the regression model successfully produced an output.
    loggin.info("doRegression - Prediction on test set completed.")


def doDeepLearning():
    # Location 15: Logging the start of the deep learning process.
    # Justification: Signifies the start of the most resource-intensive task, providing a clear marker in the logs.
    logging.info("doDeepLearning - Initiating deep learning task with MNIST dataset.")

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()


    train_images = (train_images / 255) - 0.5
    test_images = (test_images / 255) - 0.5

    # Location 16: Logging a critical data preprocessing step.
    # Justification: Normalization is vital for model performance. This logs confirms the operation was performed, which is important for ruling out preprocessing errors during an investigation.
    logging.info("doDeepLearning - Train and test images normalized successfully.")

    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    num_filters = 8
    filter_size = 3
    pool_size = 2

    model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation='softmax'),
    ])

    # Location 17: Logging the model architecture parameters.
    # Justification: Trakcs the neutral network's structure. Any deviation from this logged structure could indicate model tampering.
    logging.info(f"doDeepLearning - CNN model created with {num_filters} filters of size {filter_size}.")

    # Compile the model.
    model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

    # Location 18: Logging the compilation settings.
    # Justification: The optimizer and loss function are fundamental to training. This log records the settings, which can be checked if the model's performance becomes erratic. 
    logging.info("doDeepLearning - Model compiled with optimizer 'adam' and loss 'categorical_crossentropy'.")

    # Train the model.
    history = model.fit(
        train_images,
        to_categorical(train_labels),
        epochs=3,
        validation_data=(test_images, to_categorical(test_labels)),
    )

    # Location 19: Logging the final training and validation accuracy.
    # Justification: This is a direct measure against 'model tricking'. A sudden drop in validation accuracy is a strong indicator of an attack or other issue.
    final_val_accuracy = history.history['val_accuracy'][-1]
    logging.info("doDeepLearning - Model weights saved to cnn.h5.")
    
    model.save_weights('cnn.h5')

    # Location 20: Logging the event of saving model weights.
    # Justification: Creates an audit trail for when the authoritative model file is created or overwritten. This is key for version control and rollback in case of a compromise.
    logging.info("doDeepLearning - Model weights saved to cnn.h5.")

    predictions = model.predict(test_images[:5])

    print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

    print(test_labels[:5]) # [7, 2, 1, 0, 4]

if __name__=='__main__': 
    data_frame = readData()
    makePrediction() 
    doRegression() 
    doDeepLearning() 
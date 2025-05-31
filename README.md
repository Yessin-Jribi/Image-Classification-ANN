## Image Recognition with ANN – Fashion Item Classification

This project implements an Artificial Neural Network (ANN) to classify fashion items from the Fashion MNIST dataset. It uses TensorFlow/Keras and basic data visualization tools to train, evaluate, and analyze a multi-class image classification model.

**Dataset**

The project uses Fashion MNIST, a dataset consisting of 70,000 grayscale images (28x28 pixels) of fashion products, divided into:

fashion-mnist_train.csv – 60,000 training samples

fashion-mnist_test.csv – 10,000 test samples

To access the dataset, check this link: https://drive.google.com/file/d/1VkGDXm9hNwGSmlFmbj5mR5c6X9N99sqD/view?usp=drive_link

Each image is labeled with an integer (0–9), representing:

Label	Class

0	T-shirt/top

1	Trouser

2	Pullover

3	Dress

4	Coat

5	Sandal

6	Shirt

7	Sneaker

8	Bag

9	Ankle boot

**Project Workflow**

*1. Data Preparation*

Load datasets from CSV

Visualize a few sample images

Normalize pixel values to [0, 1]

Split features and labels

*2. Exploratory Data Analysis*

Check class distribution with a bar plot (balanced dataset)

*3. Model Building*

Constructed an ANN with:

Two hidden layers (128 neurons each, ReLU activation)

Output layer (10 neurons, Softmax)

Compiled using:

Optimizer: SGD

Loss: sparse_categorical_crossentropy

Trained for 25 epochs

*4. Evaluation*

Test accuracy: ~89%

Confusion matrix to visualize class-wise prediction performance

*5. Error Analysis*

Visualized first 9 misclassified images

Analyzed model’s confusion between similar items (e.g., shirts vs pullovers)

**Model Performance**

Good performance on most classes

Model struggles with visually similar classes (e.g., pullover vs shirt)

Confusion matrix and image visualizations help diagnose issues

**Dependencies**

Python 3

TensorFlow / Keras

Pandas

Numpy

Matplotlib

Seaborn

scikit-learn

**Future Improvements**

Tune hyperparameters (activation functions, optimizers, batch size)

Improve image quality or resolution

Try Convolutional Neural Networks (CNNs) for better visual understanding

Perform data augmentation to enhance generalization


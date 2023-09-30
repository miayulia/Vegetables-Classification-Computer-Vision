# Vegetables-Classification-Computer-Vision
A computer vision project  utilizing deep learning algorithms to classify vegetables images with a 99% accuracy score accurately. The dataset used is Vegetables Images Dataset from Kaggle.

## Project Description

Computer vision is one of the core fields of artificial intelligence aimed at teaching computers to understand and interpret the visual world. In this project, we focus on the classification of vegetable images. Our main objective is to develop a deep learning model capable of recognizing various types of vegetables with extremely high accuracy.

## Dataset

We used the Vegetables Images dataset available on Kaggle. This dataset contains thousands of color images of various types of vegetables. We divided the dataset into two parts: the training data (to train the model) and the testing data (to evaluate the model's performance). The dataset is highly diverse and includes various lighting conditions, backgrounds, and angles.

Dataset link: [Kaggle - Vegetable Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset*)

## Algorithms and Models

We employed various deep learning techniques and algorithms to develop our vegetable image classification model. Some key components of this project include:

- Convolutional Neural Networks (CNNs): We used CNNs to extract features from the images.
- Transfer Learning: We tried pre-trained architectures, such as VGG16, to leverage prior knowledge learned by these models.

## Evaluation

Our model was tested using a separate testing dataset. The model achieved an accuracy of 99%, demonstrating its ability to recognize various types of vegetables exceptionally well. We also conducted additional evaluations using metrics such as confusion matrices and ROC curves.

## Conclusion
To create a prediction-classification model for vegetables (Bean, Bitter_Gourd, Bottle_Gourd Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, and Tomato), two different models namely `Sequential` and `VGG16` were tested with custom layers added for model optimization.
- In the `sequential` model, it was found that the model has an accuracy of 85% and has a loss value of 46%. Of the 3 random images selected for prediction, the model can correctly predict 2 images.
- While in the `VGG16` model, the accuracy result is 99% and the loss value is 5%. From the same 3 random images for prediction in the sequential model, the model can correctly predict all images.
- So it is found that the `VGG16` model is the best model because it has the highest accuracy and lowest loss. Both models have the same gradient problem which is exploding. In the VGG16 model, batchnormalization has been added to prevent this. so that for model improvement it is necessary to add normalization in other ways to get optimal results.

Then, making efficiency by reducing human resources and replacing them with machines trained with deep learning models is a good thing to do. But please note that to select vegetables, the model takes a little longer than humans in general. So that several machines are needed for time efficiency.

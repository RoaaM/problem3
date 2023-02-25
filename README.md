# C# and Microsoft.ML library for ML tasks.

This project demonstrates how to use transfer learning to train an image classification model using the Inception v3 deep neural network architecture based on ImageNet weights.

The code first loads image data and labels from a CSV file using the ML.NET library, and then defines an ML pipeline consisting of several data transformation steps and the LbfgsMaximumEntropy trainer for multi-class classification. The pipeline transforms the image data by resizing and preprocessing it, and then loads the pre-trained Inception v3 model to extract features from the images.

The trained model is used to make predictions on new images, including both batch and single image predictions. The code also evaluates the model's performance by calculating the log loss and per-class log loss metrics.

Overall, this code shows how transfer learning can be used to build a high-performing image classification model with minimal training data by leveraging the pre-trained Inception v3 model.

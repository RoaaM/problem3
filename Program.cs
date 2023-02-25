using Microsoft.ML;
using System;
using System.IO;
using System.Linq;

namespace TransferLearning
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create a new MLContext instance
            var context = new MLContext();

            // Load the data from the CSV file into an IDataView
            var data = context.Data.LoadFromTextFile<ImageData>("./labels.csv", separatorChar: ',');

            // Preview the first few rows of the data
            var preview = data.Preview();

            // Define the ML pipeline
            /* in the following lines I define the ML pipeline that will be used for training the model. The pipeline consists of a series of
             * data transformations and machine learning algorithms, applied in sequence.
             
             1- 'MapValueToKey' transformation, which converts the label column to a key type. Then, it loads the images from disk using the LoadImages.
             2- transformation and resizes them to a fixed size using the 'ResizeImages' transformation.
             3- The 'ExtractPixels' transformation extracts pixel values from the images and applies preprocessing 
                steps such as interleaving pixel colors and offsetting the image by its mean.
             4- Next, the pipeline loads the Inception v3 model, which has been pre-trained on a large image dataset,
                using the 'LoadTensorFlowModel' method. ----> It scores the input images using the Inception v3 model and
                                                              extracts the softmax2_pre_activation output, which is used as the model's prediction.
             5- The 'LbfgsMaximumEntropy' trainer is used to train the model, which is then mapped back to the original 
                label values using the 'MapKeyToValue' transformation.
            
             */

            var pipeline = context.Transforms.Conversion.MapValueToKey("LabelKey", "Label")
                .Append(context.Transforms.LoadImages("input", "images", nameof(ImageData.ImagePath)))
                .Append(context.Transforms.ResizeImages("input", InceptionSettings.IMageWidth, InceptionSettings.ImageHeight, "input"))
                .Append(context.Transforms.ExtractPixels("input", interleavePixelColors: InceptionSettings.ChannelsList,
                    offsetImage: InceptionSettings.Mean))
                .Append(context.Model.LoadTensorFlowModel("C:\\Users\\roaas\\Desktop\\TransfereLearning\\TransfereLearning\\model\\tensorflow_inception_graph.pb")
                    .ScoreTensorFlowModel(new[] { "softmax2_pre_activation" }, new[] { "input" }, addBatchDimensionInput: true))
                .Append(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy("LabelKey", "softmax2_pre_activation"))
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"));

            // Train the model
            var model = pipeline.Fit(data);

            // Load the images to make predictions on
            var imageData = File.ReadAllLines("./labels.csv")
                .Select(l => l.Split(','))
                .Select(l => new ImageData { ImagePath = Path.Combine(Environment.CurrentDirectory, "images", l[0]) });

            // Convert the images to an IDataView
            var imageDataView = context.Data.LoadFromEnumerable(imageData);

            // Make predictions on the images
            var predictions = model.Transform(imageDataView);

            // Convert the predictions to an enumerable
            var imagePredictions = context.Data.CreateEnumerable<ImagePrediction>(predictions, reuseRowObject: false, ignoreMissingColumns: true);

            // Evaluate
            Console.WriteLine("\n------------Evaluate-----------------");

            // Make predictions on the training data to evaluate the model
            var evalPredictions = model.Transform(data);

            // Calculate the evaluation metrics
            var metrics = context.MulticlassClassification.Evaluate(evalPredictions, labelColumnName: "LabelKey",
                predictedLabelColumnName: "PredictedLabel");

            // Log loss should be close to 0 for accurate predictions
            Console.WriteLine($"Log Loss - {metrics.LogLoss}");
            Console.WriteLine($"Per class Log Loss - {String.Join(',', metrics.PerClassLogLoss.Select(l => l.ToString()))}");

            // Make batch predictions and print the results
            Console.WriteLine("\n------------Batch predictions-----------------");

            foreach (var prediction in imagePredictions)
            {
                Console.WriteLine($"Image - {prediction.ImagePath} is predicted as {prediction.PredictedLabelValue} " +
                    $"with a score of {prediction.Score.Max()}");
            }

            // Make a single prediction and print the result
            var predictionFunc = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

            var singlePrediction = predictionFunc.Predict(new ImageData
            {
                ImagePath = Path.Combine(Environment.CurrentDirectory, "images", "cup2.jpg")
            });

            Console.WriteLine("\n------------Single prediction-----------------");
            Console.WriteLine($"Image {Path.GetFileName(singlePrediction.ImagePath)} was predicted as a {singlePrediction.PredictedLabelValue} " +
                $"with a score of {singlePrediction.Score.Max()}");

            Console.ReadLine();
        }
    }
}
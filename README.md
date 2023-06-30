# NCVPRIPG_23
Summer Challenge Writer Verification 


## Dependencies
The code requires the following dependencies:
- torch
- torch.nn
- torch.utils.data
- PIL
- pandas
- sklearn.metrics
- os (implicitly used)

Make sure to have these dependencies installed before running the code.

## Training
1. Create data loaders: Two data loaders are created, one for the training dataset (`train_loader`) and one for the validation dataset (`val_loader`).
2. Create an instance of the model: An instance of the `WriterClassifier` model is created and moved to the appropriate device (e.g., GPU).
3. Define the loss function and optimizer: The cross-entropy loss function and Adam optimizer are defined for training the model.
4. Train the model: The model is trained for a specified number of epochs using the `train` function. The `train` function takes the model, data loader, loss function, optimizer, and the number of epochs as arguments.

## Evaluation
1. Set the model to evaluation mode: The model is set to evaluation mode using `model.eval()`.
2. Iterate over the validation data: The code iterates over the validation data loader (`val_loader`) and performs the following steps:
   - Move inputs and labels to the appropriate device.
   - Forward pass: The inputs are passed through the model to obtain the outputs.
   - Calculate predictions: The predicted labels are obtained by selecting the class with the highest probability from the output using `torch.max`.
   - Store predictions and true labels: The predictions and true labels are stored for later evaluation.

## Evaluation Metrics
- Accuracy: The accuracy of the model is calculated using `accuracy_score` from `sklearn.metrics`.
- ROC AUC Score: The ROC AUC score is calculated using `roc_auc_score` from `sklearn.metrics`.
- F1 Score: The F1 score is calculated using `f1_score` from `sklearn.metrics`.

## Test Set Prediction
1. Create a custom dataset class for the test set: The `WriterTestDataset` class is defined to load the image pairs and labels from a test CSV file and image folder. The class takes the CSV file path, root directory, and an optional transform as inputs.
2. Create test dataset and data loader: The test dataset is created using the `WriterTestDataset` class with the test CSV file path, root directory, and transform. The test data loader (`test_loader`) is then created.
3. Generate predictions for the test set: The model is set to evaluation mode, and predictions are generated for the test set using the test data loader. The probabilities are obtained by applying the softmax function to the outputs and selecting the probability for class 1 (same writer).
4. Create a submission file: The predictions are combined with the corresponding IDs from the test CSV file to create a submission dataframe.
5. Save the submission file: The submission dataframe, containing the IDs and probabilities, is saved as a CSV file named `submission.csv`.

## Usage
1. Ensure that the required dependencies are installed.
2. Update the paths to the data files and image folders:
   - `train_csv_file`: Path to the training CSV file.
   - `val_csv_file`: Path to the validation CSV file.
   - `train_root_dir`: Path to the root directory of the training images.
   - `val_root_dir`: Path to the root directory of the validation images.
   - `test_csv_file`: Path to the test CSV file.
   - `test_root_dir`: Path to

 the root directory of the test images.
3. Run the code to train the model, evaluate it on the validation set, and generate predictions for the test set.
4. The validation accuracy, AUC score, and F1 score will be printed.
5. The submission file (`submission.csv`) will be saved in the current directory.

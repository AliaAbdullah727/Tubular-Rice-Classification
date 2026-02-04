# Rice Type Classification using PyTorch

## Project Description
This project implements a simple neural network in PyTorch to classify different types of rice based on their physical characteristics. The goal is to build a binary classification model that can accurately distinguish between two classes of rice.

## Dataset
The dataset used for this classification task is the 'Rice Type Classification' dataset from Kaggle. It contains various features extracted from rice grains, such as Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, EquivDiameter, Extent, Perimeter, Roundness, and AspectRation. The target variable is 'Class', indicating the type of rice.

### Data Preprocessing
1.  **Loading Data**: The dataset is loaded using pandas from a CSV file.
2.  **Handling Missing Values**: Checked for null values, and none were found.
3.  **Feature Selection**: The 'id' column was dropped as it's not relevant for classification.
4.  **Normalization**: All numerical features were normalized by dividing by their absolute maximum value to scale them between 0 and 1.
5.  **Data Splitting**: The data was split into training, validation, and testing sets using `sklearn.model_selection.train_test_split` with a 70/15/15 ratio.

## Model Architecture
A simple feed-forward neural network (Multi-Layer Perceptron) was implemented using PyTorch. The model consists of:
-   An input layer with 10 features (corresponding to the dataset's features).
-   A hidden layer with 10 neurons.
-   An output layer with 1 neuron for binary classification.
-   A Sigmoid activation function in the output layer to output probabilities.

```python
class Mymodel(nn.Module):
  def __init__(self, hidden_layers=10):
    super().__init__()
    self.input_layer=nn.Linear(in_features=x.shape[1], out_features=hidden_layers)
    self.linear_layer=nn.Linear(in_features=hidden_layers , out_features=1)
    self.sigmoid=nn.Sigmoid()

  def forward (self,x):
    x=self.input_layer(x)
    x=self.linear_layer(x)
    x=self.sigmoid(x)
    return x
```

## Training and Evaluation
-   **Loss Function**: Binary Cross-Entropy Loss (`nn.BCELoss`) was used as the loss function, suitable for binary classification tasks.
-   **Optimizer**: Adam optimizer (`torch.optim.Adam`) with a learning rate of 0.1 was used.
-   **Epochs**: The model was trained for 10 epochs.
-   **Batch Size**: A batch size of 8 was used for training and validation.
-   **Metrics**: Training and validation loss and accuracy were tracked per epoch.

### Training Progress
The model showed consistent improvement over 10 epochs:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| :---: | :--------: | :-------: | :------: | :-----: |
|   0   |   0.0952   |  0.9786   |  0.0875  |  0.9791 |
|   1   |   0.0595   |  0.9837   |  0.0477  |  0.9857 |
|   2   |   0.1128   |  0.9824   |  0.0592  |  0.9861 |
|   3   |   0.2097   |  0.9797   |  0.0986  |  0.9842 |
|   4   |   0.0710   |  0.9830   |  0.0485  |  0.9857 |
|   5   |   0.0757   |  0.9809   |  0.0452  |  0.9868 |
|   6   |   0.1065   |  0.9807   |  0.0988  |  0.9861 |
|   7   |   0.1054   |  0.9834   |  0.0438  |  0.9857 |
|   8   |   0.1193   |  0.9825   |  0.0634  |  0.9776 |
|   9   |   0.1138   |  0.9785   |  0.0503  |  0.9831 |

## Results
After training, the model was evaluated on the unseen test set:

-   **Test Loss**: `0.0272`
-   **Test Accuracy**: `0.9897`

The model demonstrates excellent performance on the test set, achieving nearly 99% accuracy, indicating strong generalization capabilities.

## How to Run
1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <Tubular_Rice_Classification>
    ```
2.  **Install Dependencies**:
    Ensure you have Python installed. Then, install the necessary libraries:
    ```bash
    pip install torch torchvision scikit-learn pandas matplotlib opendatasets
    ```
3.  **Download the Dataset**:
    The notebook uses `opendatasets` to download the Kaggle dataset. Make sure you have a `kaggle.json` file with your credentials configured, or manually download the `rice-type-classification.zip` from [Kaggle](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification) and place `riceClassification.csv` in a `/content/rice-type-classification/` directory.

4.  **Run the Notebook**:
    Open the Jupyter Notebook (`.ipynb` file) in your preferred environment (e.g., Jupyter Lab, Google Colab) and run all cells.

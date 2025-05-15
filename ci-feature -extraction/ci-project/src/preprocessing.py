import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    A class to handle data preprocessing steps.
    """
    def __init__(self, target_column, test_size=0.2, random_state=42):
        """
        Initializes the DataPreprocessor.

        Args:
            target_column (str): The name of the target variable column.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Seed used by the random number generator.
        """
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.X = None
        self.y = None

    def preprocess(self, df):
        """
        Performs preprocessing steps: separating features and target,
        scaling features (fitting only on training data), and splitting data.

        Args:
            df (pandas.DataFrame): The input DataFrame.

        Returns:
            tuple: A tuple containing:
                - X_train_scaled (np.ndarray): Scaled training features.
                - X_test_scaled (np.ndarray): Scaled testing features.
                - y_train (pandas.Series): Training target.
                - y_test (pandas.Series): Testing target.
                - feature_names (list): List of feature names.
        """
        if df is None:
            raise ValueError("Input DataFrame is None.")

        self.X = df.drop(columns=[self.target_column])
        self.y = df[self.target_column]
        feature_names = self.X.columns.tolist()

        # Split data BEFORE scaling
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, stratify=self.y, random_state=self.random_state
        )

        # Fit scaler ONLY on the training data and transform both
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names
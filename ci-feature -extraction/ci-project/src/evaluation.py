from sklearn.metrics import accuracy_score
# Assuming the model (e.g., LogisticRegression) is passed during initialization
# from sklearn.linear_model import LogisticRegression # Example import

class ModelEvaluator:
    """
    A class to train and evaluate a classification model.
    """
    def __init__(self, model):
        """
        Initializes the ModelEvaluator with a classification model.

        Args:
            model: An unfitted classification model object with fit and predict methods.
        """
        self.model = model

    def evaluate(self, X_train, X_test, y_train, y_test):
        """
        Trains the model on the training data and evaluates its accuracy on the test data.

        Args:
            X_train (np.ndarray): Training features (dimensionality-reduced).
            X_test (np.ndarray): Testing features (dimensionality-reduced).
            y_train (pandas.Series): Training target.
            y_test (pandas.Series): Testing target.

        Returns:
            float: The accuracy score on the test data.

        Raises:
            ValueError: If input data is None.
            Exception: For errors during model training or prediction.
        """
        if X_train is None or X_test is None or y_train is None or y_test is None:
             raise ValueError("Input data for evaluation cannot be None.")

        try:
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            return accuracy_score(y_test, preds)
        except Exception as e:
             raise Exception(f"Error during model evaluation: {e}")
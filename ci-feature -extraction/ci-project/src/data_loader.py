import pandas as pd
import os
class DataLoader:
    """
    A class to handle dataset loading.
    """
    def __init__(self, data_path):
        """
        Initializes the DataLoader with the path to the dataset.

        Args:
            data_path (str): The full path to the dataset CSV file.
        """
        self.data_path = data_path

    def load_data(self):
        """
        Loads the dataset from the specified CSV file.

        Returns:
            pandas.DataFrame: The loaded DataFrame.

        Raises:
            FileNotFoundError: If the data file does not exist.
            Exception: For other errors during loading.
        """
        print(f"Attempting to load data from: {self.data_path}")
        if not os.path.exists(self.data_path):
             raise FileNotFoundError(f"Data file not found at: {self.data_path}")
        try:
            df = pd.read_csv(self.data_path)
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def get_info(self, df):
        """
        Prints concise summary of a DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to summarize.
        """
        df.info()

    def get_description(self, df):
        """
        Generates descriptive statistics of a DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to describe.

        Returns:
            pandas.DataFrame: Descriptive statistics.
        """
        return df.describe()

    def get_value_counts(self, df, column):
        """
        Returns a Series containing counts of unique values in a column.

        Args:
            df (pandas.DataFrame): The DataFrame.
            column (str): The column name.

        Returns:
            pandas.Series: Value counts.
        """
        return df[column].value_counts()
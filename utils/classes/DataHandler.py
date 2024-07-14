import pandas as pd
from datetime import datetime
import os 
import json

class DataHandler:
    """
    This class provides functionality to save data structures to various file formats.

    The data structure is first converted to a Pandas DataFrame if necessary, and then
    saved to a file in the specified format (CSV, JSON, or pickle).

    Attributes:
        structure: The data structure to be saved.

    Methods:
        save_data: Saves the data structure to a file.
        load_file: Saves the data structure from a file.

    """
    def __init__(self, structure):
        self.data_structure = structure

    def save_data(self,  filename, format="csv", folder="data/raw_data", data_structure=None):
        """
        Saves the data structure to a file.

        Args:
            format (str, optional): The desired file format. Defaults to "csv".
                Supported formats are "csv", "json", and "txt".
            filename (str, optional): The base name of the output file (without extension).
                Defaults to "posts_data".

        Raises:
            ValueError: If an invalid format is provided.
        """
        if not (data_structure):
            posts_data = self.data_structure
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Add a timestamp to the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{folder}/{timestamp}_{filename}"
        if format == "csv":

            if not isinstance(posts_data, pd.DataFrame):
                try: 
                    posts_data = pd.DataFrame(posts_data)
                except Exception as e:
                    print(f"[Error]: {e}")
            posts_data.to_csv(f"{filename}.csv", index=False)
        elif format == "json":
            # Convert datetime objects to strings
            for post in posts_data:
                if isinstance(post['post_created_utc'], datetime):  # Check if it's a datetime
                    post['post_created_utc'] = post['post_created_utc'].isoformat()

            with open(filename, 'w') as f:
                json.dump(posts_data, f)


        else:
            raise ValueError("Invalid format. Choose from 'csv' or 'json'.")
        
    def load_file(format="csv", filename="posts_data", folder="data/raw_data"):
        """
        Loads post data from a file into a Pandas DataFrame.

        Args:
            format (str, optional): The file format to load. Defaults to "csv".
                Supported formats: "csv", "json", "pickle".
            filename (str, optional): The base filename (without extension). Defaults to "posts_data".
            folder (str, optional): The directory containing the file. Defaults to "data/raw_data".

        Returns:
            pandas.DataFrame: The loaded post data.

        Raises:
            ValueError: If an invalid format is specified.
        """
        filename = f"{folder}/{filename}"

        if format == "csv":
            return pd.read_csv(f"{filename}.csv")
        elif format == "json":
            return pd.read_json(f"{filename}.json", orient="records")
        elif format == "pickle":
            return pd.read_pickle(f"{filename}.pkl")
        else:
            raise ValueError("Invalid format. Choose from 'csv', 'json', or 'pickle'.")

import praw 
from datetime import datetime
import pandas as pd
from utils.classes.DataHandler import DataHandler


class SubredditsDataCollector(DataHandler):
    def __init__(self, reddit_instance: praw.Reddit, data_structure=None):
        self.reddit = reddit_instance  # Store the PRAW Reddit instance
        self.subreddit_counts = {}
        self.private_subreddit_list = []
        self.data_structure = data_structure if data_structure else pd.DataFrame()


    def search_subreddits(self, search_terms, excluded_subreddits=[], limit=None):
        """
        Searches for relevant subreddits and initializes a counter for each.

        This function performs the following steps:
        
        1. Iterates through a list of search terms.
        2. For each term, uses PRAW's `reddit.subreddits.search()` to find matching subreddits.
        3. Filters out any subreddits whose names are in the `excluded_subreddits` list.
        4. Creates a dictionary where keys are the names of the selected subreddits and values
        are initialized to 0. This dictionary will later be used to count mentions.
        
        Args:
            reddit (praw.Reddit): An authenticated PRAW Reddit instance.
            search_terms (list): A list of strings representing terms to search for in subreddit descriptions.
            excluded_subreddits (list): A list of strings representing subreddit names to exclude.
            limit (int, optional): The maximum number of subreddits to return per search term. Defaults to None (no limit).

        Returns:
            None
        """
        for term in search_terms:
            for subreddit in self.reddit.subreddits.search(term, limit=limit):
                if subreddit.display_name.lower() not in excluded_subreddits:
                    self.subreddit_counts[subreddit.display_name] = 0
        print(f"Selected subreddits: {len(self.subreddit_counts)}")



    def filter_subreddits(self, top_n=None, subreddits=None):
        """
        Filters a dictionary of subreddit counts based on either the top N subreddits or a specific list.

        Args:
            top_n (int, optional): The number of top subreddits to return (default: None).
            subreddits (list, optional): A list of subreddit names to include (default: None).

        Returns:
            dataframe: A dictionary containing the filtered subreddits and their counts.

        Raises:
            ValueError: If both top_n and specific_subreddits are provided or if neither is provided.

        """
        if top_n and subreddits:
            raise ValueError("[Error] Cannot specify both top_n and subreddits")
        if not top_n and not subreddits:
            raise ValueError("[Error] Must specify either top_n or subreddits")

        if top_n:
            self.subreddit_counts =  dict(
                sorted(
                    self.subreddit_counts.items(), key=lambda item: item[1], reverse=True
                )[:top_n]
            )
        else:
            self.subreddit_counts = {
                subreddit: count
                for subreddit, count in self.subreddit_counts.items()
                if subreddit in subreddits
            }

        if not self.data_structure.empty: 
            self.data_structure = self.data_structure[self.data_structure['subreddit'].isin(self.subreddit_counts.keys())]


        return self.data_structure 
    


    def get_subreddit_details(self):
        """
        Fetches various information about subreddits from Reddit using the PRAW library.

        Args:
            reddit (praw.Reddit): An authenticated PRAW Reddit instance.
            subreddits (list): A list of subreddit names to fetch data for.

        Returns:
            pandas.DataFrame: A DataFrame containing the following columns:
                - subreddit: The name of the subreddit.
                - created_utc: The date and time the subreddit was created (in UTC).
                - subscribers: The number of subreddit subscribers.
                - description: The public description of the subreddit.
                - over18: Whether the subreddit is marked as Not Safe For Work (NSFW).

        Raises:
            praw.exceptions.RedditAPIException: If there's an error communicating with the Reddit API (e.g., invalid subreddit name).

        """
        
        all_data = []
        subreddits = self.subreddit_counts.keys()
        for subreddit_name in subreddits:
            # print(f"Fetching data for subreddit: {subreddit_name}")

            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                all_data.append({
                    "subreddit": subreddit.display_name,
                    "created_utc": datetime.utcfromtimestamp(subreddit.created_utc).date(),
                    "subscribers": subreddit.subscribers,
                    "description": subreddit.public_description or "",
                    "over18": subreddit.over18,
                })

            except Exception as e:
                if str(e) == "received 403 HTTP response":
                    print(f"[Error] Access to subreddit '{subreddit_name}' forbidden: {e}")
                    self.private_subreddit_list.append(subreddit_name)
                    continue  # Skip this subreddit if it's private
            
                else:
                    print(f"[Error] Fetching data for subreddit: {subreddit_name}, Error: {e}")
        
        self.delete_subreddits()
        
        
        self.data_structure = pd.DataFrame(all_data).sort_values(by=['subscribers'], ascending=False)
        self.save_subreddit_data()
        return self.data_structure


    def delete_subreddits(self, private_subreddit_list = []):
        """
        Removes subreddits .
        """
        private_subreddit_list = private_subreddit_list + self.private_subreddit_list

        for key in private_subreddit_list:

            self.subreddit_counts.pop(key, None)
    
        if not self.data_structure.empty:
            self.data_structure = self.data_structure[~self.data_structure['subreddit'].isin(private_subreddit_list)] 



    def save_subreddit_data(self,format="csv", folder="data", filename="subreddits_details"):  
        """
        Saves the collected subreddit details to a file.

        This method takes the subreddit details stored in the `self.data_structure`
        attribute and writes them to a file. It handles the creation of the specified folder 
        if it doesn't exist and adds a timestamp to the filename to avoid overwriting existing files.

        Args:
            folder (str, optional): The folder where the a file will be saved. Defaults to "data".
            filename (str, optional): The base name of the a file (without the timestamp).
                Defaults to "subreddits_details.csv".

        Returns:
            None
        """
        filename = f"{filename}"
        super().save_data(format=format, folder=folder, filename=filename)

    def load_subreddit_data(self, filename, format="csv", folder="data/raw_data"):  
        """
        Loads the collected subreddit data from a file.

        This method takes the subreddit data stored in a file and load them to a `self.data_structure`


        Args:
            folder (str, optional): The folder where the a file will be saved. Defaults to "data".
            filename (str, optional): The base name of the a file.

        Returns:
            None
        """
        self.data_structure = super().save_data(filename=filename, format=format, folder=folder)
        return self.data_structure

    def add_post_counts(self, id, column_name='num_posts'):
        """Adds a 'num_posts' column to the dataframe based on subreddit counts.

        Args:
            df (pd.DataFrame): The input dataframe with a 'subreddit' column.
            sorted_subreddits (list): A list of tuples containing subreddit names and their post counts.

        Returns:
            pd.DataFrame: The modified dataframe with the added 'num_posts' column.
        """
        sorted_subreddits = sorted(self.subreddit_counts.items(), key=lambda x: x[1], reverse=True)
        column_name = f"{id}_{column_name}"
        subreddit_counts = dict(sorted_subreddits)
        self.data_structure[column_name] = self.data_structure.loc[:,'subreddit'].map(subreddit_counts).fillna(0)
        # self.data_structure[column_name] = self.data_structure.loc[:,column_name].astype(int) 
        return self.data_structure
  
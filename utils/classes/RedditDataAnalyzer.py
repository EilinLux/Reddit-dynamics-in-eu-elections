

from utils.classes.RedditPostsCollector import RedditPostsCollector
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



class RedditDataAnalyzer(RedditPostsCollector):
    """
    Analyzes and Visualizes posts and comments from Reddit subreddits.

 
    """

    def __init__(self, post_collector = None, file_name: str=None, child_dataanalyzer=None):
        if post_collector:
            self.data_structure = post_collector.data_structure
        if file_name:
            self.data_structure = self.load_posts_data(file_name)
        if child_dataanalyzer:
            self.data_structure = child_dataanalyzer

    def load_posts_data(self, filename="posts_data.txt"):
        """
        
        filename="data/raw_data/20240712_101844_posts_data_full_posts_data" 

        """
        with open(filename, 'r') as f:
            posts_data = json.load(f)

            # Convert datetime strings back to datetime objects
            for post in posts_data:
                if not isinstance(post['post_created_utc'], datetime):  # Check if it's NOT a datetime
                    post['post_created_utc'] = datetime.fromisoformat(post['post_created_utc'])

            return posts_data
        


    def subreddit_stats(self):
        """
        Calculates and prints descriptive statistics for subreddit post data.

        Args:
            posts_data (list): List of dictionaries, each containing information about a subreddit post.
                            Expects keys: 'subreddit', 'score', 'num_comments', 'post_created_utc', 'selftext' (optional).

        Returns:
            pandas.DataFrame: A DataFrame containing the aggregated statistics for each subreddit.
        """

        df = pd.DataFrame(self.data_structure)

        stats_df = df.groupby('subreddit').agg({
            'post_id': ["count"],
            'post_sentiment': [ 'mean', 'std'],
            'comments_num': ["sum", 'mean', 'std'],
            'post_upvote_score': ["mean", 'std'],
            'post_upvote_ratio': ["mean", 'std'],
        })


        stats_df.columns = ['_'.join(col) for col in stats_df.columns.values]  # Flatten column names
        

        stats_df['comments_sentiment_std_dev_mean'] = df.groupby('subreddit')['comments_sentiment_std_dev'].mean()
        stats_df['comments_sentiment_std_dev_std_dev'] = df.groupby('subreddit')['comments_sentiment_std_dev'].std()

        # Print statistics
        print("\nDescriptive Statistics for Subreddits:")
        ordered_stats_df = stats_df.sort_values(by=[ "post_id_count"], ascending=False)
        print(ordered_stats_df.to_markdown(numalign="left", stralign="left"))

        return ordered_stats_df 
    



    def count_author_mentions(self, exclude_author_list, min_mentions=1, include_null_authors=False):
        """
        (This function should count author mentions in  data
        """
        subreddit_author_counts = defaultdict(lambda: defaultdict(int))  # Nested defaultdict

        for post in self.data_structure:
            subreddit = post['subreddit']

            # Count post author
            author = post['post_author']

            if author or include_null_authors:
                if author not in exclude_author_list:
                    subreddit_author_counts[subreddit][author] += 1
                   

            # Count comment authors
            for comment_author in post['comments_author']:
                if comment_author or include_null_authors:
                    subreddit_author_counts[subreddit][comment_author] += 1

        # Filter out authors with less than min_mentions
        for subreddit, author_counts in subreddit_author_counts.items():
            subreddit_author_counts[subreddit] = {
                author: count for author, count in author_counts.items() if count >= min_mentions
            }

        return subreddit_author_counts
    

    def visualize_author_subreddit_interaction_graph(self, posts_data_list=None, min_mentions=1, 
                                                include_null_authors=False, 
                                                min_subreddits_per_author=2, 
                                                exclude_author_list=[]):
        """
        Counts the number of times authors are mentioned in posts and comments, optionally filtering by
        a minimum mention threshold, whether to include authors with null/None names, and the minimum 
        number of subreddits to which an author contributes.

        Displays the Sankey diagram visualization.

        Args:
            posts_data_list (list): A list of dictionaries, each containing post data.
            min_mentions (int, optional): Minimum mentions for an author to be included. Defaults to 1.
            include_null_authors (bool, optional): Whether to include authors with null names. Defaults to False.
            min_subreddits_per_author (int, optional): Minimum number of subreddits an author must be 
                active in to be included. Defaults to 1.

        Returns:
            authors: list of authors , matching the filters
            
        """

        all_subreddits = set()
        author_color_map = {}
        if posts_data_list == None:
            posts_data_list=[self.data_structure]

        for posts_data in posts_data_list:
            if not posts_data:
                continue

            subreddit_author_counts = self.count_author_mentions(exclude_author_list, min_mentions, include_null_authors)
            
            # Filter authors based on min_subreddits_per_author
            filtered_author_counts = {}
            for subreddit, author_counts in subreddit_author_counts.items():
                filtered_author_counts[subreddit] = {
                    author: count 
                    for author, count in author_counts.items() 
                    if len({sub for sub, counts in subreddit_author_counts.items() if author in counts}) >= min_subreddits_per_author
                }

            # Update subreddit_author_counts with filtered results
            subreddit_author_counts = filtered_author_counts

            subreddits = list(subreddit_author_counts.keys())
            all_subreddits.update(subreddits)
            authors = set()
            for author_counts in subreddit_author_counts.values():
                authors.update(author_counts.keys())
            authors = list(authors)

            node_labels = subreddits + authors
            node_colors = ["skyblue"] * len(subreddits) + ["lightgray"] * len(authors)

            # Create unique colors for authors
            num_authors = len(authors)
            cmap = plt.colormaps['tab20']  # Use the new colormaps object
            norm = mcolors.Normalize(vmin=0, vmax=num_authors - 1) 

            author_color_map = {
                author: mcolors.to_hex(cmap(norm(i)))  # Convert colormap value to hex
                for i, author in enumerate(authors)
            }



            # Color author nodes
            for author in authors:
                node_colors.append(author_color_map[author])  # Use author-specific color for nodes

            # Create links and assign colors
            link_sources = []
            link_targets = []
            link_values = []
            link_colors = []
            for subreddit_index, (subreddit, author_counts) in enumerate(subreddit_author_counts.items()):
                for author, count in author_counts.items():
                    author_index = authors.index(author) + len(subreddits) 
                    link_sources.append(subreddit_index)
                    link_targets.append(author_index)
                    link_values.append(count)
                    link_colors.append(author_color_map[author])  # Use author-specific color for links

            # Create Plotly figure
            fig = go.Figure(
                data=[
                    go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=node_labels,
                            color=node_colors,
                        ),
                        link=dict(
                            source=link_sources,
                            target=link_targets,
                            value=link_values,
                            color=link_colors, 
                        ),
                    )
                ]
            )
            fig.update_layout(
                title_text="Author-Subreddit Interaction Graph",
                font_size=10,
            )

            fig.show()
            return authors


    def filter_posts_by_date(self, date_string):
        """Filters posts based on a 5-year window around the given date and returns a new DataAnalyzer object.

        Args:
            date_string (str): A date string in the format 'DD-MM-YYYY'.

        Returns:
            DataAnalyzer: A new DataAnalyzer instance with the filtered posts.
        """

        try:
            target_date = datetime.strptime(date_string, '%d-%m-%Y')
        except ValueError:
            raise ValueError("Invalid date format. Use 'DD-MM-YYYY'.")

        start_date = target_date - timedelta(days=365 * 2.5)
        end_date = target_date + timedelta(days=365 * 2.5)



        filtered_posts = []
        for post in self.data_structure:  # Filter the existing posts
            try:
                created_utc = post['post_created_utc']
                if start_date <= created_utc <= end_date:
                    filtered_posts.append(post)
            except KeyError:
                print(f"Warning: Post is missing 'created_utc' key: {post}")

        # Create a new RedditDataAnalyzer instance with the filtered data
        new_analyzer = RedditDataAnalyzer(child_dataanalyzer=filtered_posts)  
        return new_analyzer
    


        def filter_posts_by_subreddit(self, list_subreddit):
            """Filters posts based on subreddit and returns a new DataAnalyzer object.

            Args:
                list_subreddit (list): A list of subreddit names.

            Returns:
                DataAnalyzer: A new DataAnalyzer instance with the filtered posts.
            """



            filtered_posts = []
            for post in self.data_structure:  # Filter the existing posts
                try:
                    subreddit = post['subreddit']
                    if subreddit in list_subreddit:
                        filtered_posts.append(post)
                except KeyError:
                    print(f"Warning: Post is missing 'subreddit' key: {post}")

            # Create a new RedditDataAnalyzer instance with the filtered data
            new_analyzer = RedditDataAnalyzer(child_dataanalyzer=filtered_posts)  
            return new_analyzer
from datetime import datetime
from textblob import TextBlob
from statistics import mean, stdev
import langdetect  # For language detection
from utils.classes.SubredditsDataCollector import SubredditsDataCollector 
import pandas as pd
import re 
from utils.classes.DataHandler import DataHandler
import plotly.graph_objects as go
from colour import Color
import time

class RedditPostsCollector(DataHandler):
    """
    Collects and analyzes posts and comments from Reddit subreddits.

    This class leverages the functionality provided by `SubredditsDataCollector`
    to retrieve information about subreddits and their posts. It then performs
    analysis on the posts and comments, extracting relevant data like sentiment,
    content length, and author information.

    Attributes:
        reddit (praw.Reddit): A PRAW Reddit instance.
        subreddit_counts (dict): A dictionary mapping subreddit names to their mention counts.
        data_structure (list): A list to store the collected and analyzed post data.
    """

    def __init__(self, subreddit_datacollector: SubredditsDataCollector, id:str, data_structure = []):
        """
        Initializes the RedditPostsCollector instance.

        This constructor establishes the necessary connections and data structures for 
        collecting posts from Reddit. It initializes attributes that store the PRAW Reddit 
        instance, subreddit information, and the structure for accumulating collected data.

        Args:
            subreddit_datacollector (SubredditsDataCollector): An instance of 
                `SubredditsDataCollector` to retrieve subreddit metadata and PRAW instance.
            id (str): A unique identifier for this data collection instance.
            data_structure (list, optional): An optional initial data structure (e.g., list)
                to store the collected post data. If not provided, an empty list is created.
        """
        self.subreddit_datacollector = subreddit_datacollector
        self.reddit = subreddit_datacollector.reddit
        self.subreddit_counts = subreddit_datacollector.subreddit_counts
        self.data_structure = data_structure if data_structure else []
        self.id = id 
        self.post_ids_seen = set()  # Create a set to track seen post IDs (for duplicate detection)
        self.subreddit_ids_seen = set() # Create a set to track seen rubreddits IDs (for multiple launching)


    def extract_subreddit_posts_and_comments(
        self, 
        num_subreddits_limit=None, 
        exclude_subreddits_limit = None,
        terms=['eu election'], 
        subreddit_post_limit=1, 
        subreddit_comments_limit=3, 
        language='en', 
        start_date=None, 
        end_date=None, 
        check_comments_language=False, 
        save_on_the_fly=True,
        save_at_the_end=True,
        reset_research=False):
        """
        Extracts and analyzes posts and comments from selected subreddits.

        This method iterates over specified subreddits, searching for posts matching
        the given terms. It filters posts based on date range and language, then extracts
        post details and analyzes comments. Sentiment analysis is performed on both
        post titles and comments. The collected data is organized into a list of
        dictionaries, each representing a single post and its analyzed comments.

        Args:
            num_subreddits_limit (int, optional): Maximum number of subreddits to process.
            exclude_subreddits_limit (list, optional): Subreddits to exclude.
            terms (list, optional): List of search terms to match in post titles. Defaults to ['eu election'].
            subreddit_post_limit (int, optional): Maximum posts per subreddit per term. Defaults to 1.
            subreddit_comments_limit (int, optional): Maximum comments per post. Defaults to 3.
            language (str, optional): Language to filter posts by (ISO 639-1 code). Defaults to 'en'.
            start_date (str, optional): Start date in 'DD-MM-YYYY' format.
            end_date (str, optional): End date in 'DD-MM-YYYY' format.
            check_comments_language (bool, optional): Whether to filter comments by language.
            save_on_the_fly (bool, optional): Save data for each subreddit after processing.
            save_at_the_end (bool, optional): Save all data at the end of processing.

        Returns:
            tuple: A tuple containing:
                - list: A list of dictionaries, each representing a post and its comments:
                    - 'subreddit': (str) Subreddit name
                    - 'post_id': (str) Unique post ID
                    - 'post_title': (str) Post title
                    - 'post_author': (str) Post author's username
                    - 'post_text': (str) Post body text
                    - 'post_sentiment': (float) Sentiment score of the post title and text
                    - 'post_created_utc': (datetime) Post creation time (UTC)
                    - 'post_upvote_score': (int) Number of upvotes for the submission
                    - 'post_upvote_ratio': (float) Percentage of upvotes from all votes
                    - 'post_url': (str) URL of the submission
                    - 'comments_sentiment': (list) Sentiment scores of each comment
                    - 'comments_bodies': (list) Text content of each comment
                    - 'comments_lenght': (list) Length of each comment (character count)
                    - 'comments_author': (list) Usernames of comment authors
                    - 'comments_num': (int) Total number of comments
                    - 'comments_sentiment_mean': (float) Mean sentiment of comments
                    - 'comments_sentiment_std_dev': (float) Std deviation of comment sentiments

                - dict: A dictionary with subreddit names and the number of analyzed posts per subreddit.
        """
        private_subreddit_list = []  # Initialize a list to store private subreddits.

        # Parse start and end dates if provided
        if start_date:
            start_date = datetime.strptime(start_date, '%d-%m-%Y')  # Convert to datetime for comparison
        if end_date:
            end_date = datetime.strptime(end_date, '%d-%m-%Y')

        print('[Analyze] started ')  # Indicate analysis has begun.

        # Determine subreddits to analyze
        if exclude_subreddits_limit:  # If a list of subreddits to exclude is provided...
            subreddits_to_analyze = {  # Create a dictionary of subreddits to analyze
                subreddit: count 
                for subreddit, count in self.subreddit_counts.items()  # ...from the available counts
                if subreddit not in exclude_subreddits_limit  # ...excluding those in the exclusion list.
            }
        else:  # Otherwise, analyze all subreddits in the counts
            subreddits_to_analyze = self.subreddit_counts 

        # Limit the number of subreddits to analyze if a limit is provided
        if num_subreddits_limit:
            subreddits_to_analyze = dict(list(subreddits_to_analyze.items())[:num_subreddits_limit])
        
        
        if reset_research:
            self.post_ids_seen = set()  # Create a set to track seen post IDs (for duplicate detection)
            self.subreddit_ids_seen = set()
        
        
        for subreddit_name in subreddits_to_analyze:  # Iterate over the subreddits to analyze

            # Check already visited
            if subreddit_name in self.subreddit_ids_seen:
                print(f"[Info] Skipping Already visited subreddit ID: {submission.id}")
                continue

            try:
                subreddit = self.reddit.subreddit(subreddit_name)  # Get the PRAW Subreddit object
                print(f'[Analyzing] {subreddit_name}')  # Print a progress message

                for term in terms:  # Iterate over the search terms


                    for submission in subreddit.search(term, limit=subreddit_post_limit):
                        #print(f'[Analyzing] {subreddit_name}: {submission.title}')  # (Optional) Print the post title
                        post_date = datetime.utcfromtimestamp(submission.created_utc)  # Get post creation time

                        # Apply date filters
                        if start_date and post_date < start_date:  # Skip if post is too early
                            continue
                        if end_date and post_date > end_date:  # Skip if post is too late
                            continue

                        # Apply language filter and handle langdetect exceptions
                        try:
                            if langdetect.detect(submission.title) != language:  # Skip if wrong language
                                continue
                        except langdetect.lang_detect_exception.LangDetectException:
                            print(f'[Warn] Language detection failed for post ID: [{submission.id}] {submission.title}')
                            continue 

                        # Check for duplicate post ID
                        if submission.id in self.post_ids_seen:
                            print(f"[Info] Skipping duplicate post ID: {submission.id}")
                            continue


                        post_data = {
                            'subreddit': subreddit_name,
                            'post_id': submission.id,
                            'post_title': submission.title,
                            'post_author': submission.author.name,
                            'post_text': submission.selftext,
                            'post_sentiment': TextBlob(submission.title + submission.selftext).sentiment.polarity,
                            'comments_sentiment': [],
                            'comments_bodies': [],
                            'comments_lenght': [],
                            'comments_author': [],
                            'comments_num': 0,
                            'comments_sentiment_mean': 0,
                            'comments_sentiment_std_dev': 0,
                            'post_created_utc': post_date,
                            'post_upvote_score': submission.score, # The number of upvotes for the submission.
                            'post_upvote_ratio': submission.upvote_ratio, # The percentage of upvotes from all votes on the submission.
                            'post_url':  submission.url # The URL the submission links to, or the permalink if a  selfpost.
                        }


                        submission.comments.replace_more(limit=subreddit_comments_limit)
                        total_discussion_length = 0
                        comments_num = 0 
                        for comment in submission.comments.list():
                            if check_comments_language == True:
                                try:
                                    if langdetect.detect(comment.body) == language:
                                        post_data['comments_sentiment'].append(TextBlob(comment.body).sentiment.polarity)
                                        post_data['comments_bodies'].append(comment.body)
                                        post_data['comments_lenght'].append(len(comment.body))
                                        total_discussion_length = total_discussion_length + len(comment.body)
                                except langdetect.lang_detect_exception.LangDetectException:
                                    # If language detection fails for a comment, skip it
                                    print(f'[Warn] Language detection failed for comment ID: [{comment.id}] {comment.body}')
                            else:
                                post_data['comments_sentiment'].append(TextBlob(comment.body).sentiment.polarity)
                                post_data['comments_bodies'].append(comment.body)
                                post_data['comments_author'].append(comment.author.name if comment.author else None)
                                post_data['comments_lenght'].append(len(comment.body))  
                                comments_num = comments_num +1

                        post_data["comments_num"] = comments_num
                        post_data['comments_sentiment_mean'] = mean(post_data['comments_sentiment']) if post_data['comments_sentiment'] else None
                        
                        # Check for enough data points for std dev
                        if len(post_data['comments_sentiment']) >= 2:
                            post_data['comments_sentiment_std_dev'] = stdev(post_data['comments_sentiment'])
                        else:
                            post_data['comments_sentiment_std_dev'] = None  # Or any other default value (e.g., 0)

                        self.data_structure.append(post_data)
                        self.subreddit_counts[subreddit_name] += 1
                        self.post_ids_seen.add(submission.id)  # Add ID to the set for future duplicate checks

                
                if save_on_the_fly == True:
                    self.save_posts_data(filename=subreddit)
                
                self.subreddit_ids_seen.add(subreddit_name)

            except Exception as e:
                if str(e) == 'received 403 HTTP response':
                    print(f"[Error] Access to subreddit '{subreddit_name}' forbidden: {e}")
                    private_subreddit_list.append(subreddit_name)
                    continue  # Skip this subreddit if it's private

                elif str(e) == 'received 429 HTTP response':
                    print(f"[Error] Access to subreddit '{subreddit_name}' Too Many requests: {e}")
                    private_subreddit_list.append(subreddit_name)
                    time.sleep(2)
                      # Skip this subreddit if it's private
                else:
                    print(f"[Error] searching in {subreddit_name}: {e}")  # Print error message for debugging
                    self.subreddit_datacollector.delete_subreddits(private_subreddit_list)

                    if save_on_the_fly == True:
                        self.save_posts_data(filename="__error")

                    return self.data_structure, subreddits_to_analyze
        
        # self.subreddit_datacollector.add_post_counts('pre_num_posts')
        self.subreddit_datacollector.delete_subreddits(private_subreddit_list)

        if save_at_the_end == True:
            self.save_posts_data()

        return self.data_structure, subreddits_to_analyze
    

    
    def save_posts_data(self, filename="posts_data", format="json", folder="data/raw_data"):  
        """
        Saves the collected post data to a file in the specified format.

        This method persists the accumulated post data stored in the `self.data_structure` 
        attribute to a file. It leverages the functionality of the parent class's `save_data` 
        method to handle file writing. The filename is constructed to include a timestamp
        and a unique instance ID to prevent overwriting and enhance traceability.

        Args:
            filename (str, optional): The base name of the output file (without extension).
                Defaults to "posts_data".
            format (str, optional): The desired file format ("csv" or "json"). Defaults to "json".
            folder (str, optional): The directory in which to save the file. Defaults to "data/raw_data".
    
        Returns:
            None
        """

        filename = f"posts_data_{self.id}_{filename}"
        super().save_data(format=format, folder=folder, filename=filename)

    # def load_posts_data(self, filename, format="pickle", folder="data/raw_data"):  
    #     """
    #     Loads the collected post data from a file.

    #     This method takes the post data stored in a file and load them to a `self.data_structure`


    #     Args:
    #         folder (str, optional): The folder where the a file will be saved. Defaults to "data".
    #         filename (str, optional): The base name of the a file.

    #     Returns:
    #         None
    #     """
    #     self.data_structure = super().save_data(filename=filename, format=format, folder=folder)
    #     self.reddit = self.reddit

    def visualize_cumulative_stacked_area_chart(self):
        """
        Generates a cumulative stacked area chart visualizing the temporal progression
        of post counts across subreddits.

        This method constructs a cumulative stacked area chart using the collected post data.
        The x-axis represents the date (`created_utc`), and the y-axis represents the cumulative
        total count of posts within each subreddit over time. This visualization provides insights
        into the relative activity levels of different subreddits and their growth patterns.

        The method first converts the collected post data into a Pandas DataFrame for analysis.
        It then groups the data by date and subreddit, calculating the cumulative sum of post
        counts for each subreddit. Finally, the chart is rendered using Plotly, with each subreddit 
        represented as a separate stacked area.

        Returns:
            None: This method does not return a value but displays the interactive chart.
        """
        # Convert data_structure to a DataFrame
        df = pd.DataFrame(self.data_structure)

        # Convert 'post_created_utc' to datetime objects (assuming it's a string column)
        df['post_created_utc'] = pd.to_datetime(df['post_created_utc'])


        # Group by post_created_utc and subreddit, and count the number of posts
        df_grouped = df.groupby([df['post_created_utc'].dt.date, 'subreddit']).size().unstack(fill_value=0)

        # Calculate the cumulative sum over time
        df_cumulative = df_grouped.cumsum()

        # Create a stacked area chart
        fig = go.Figure()

        for subreddit in df_cumulative.columns:
            fig.add_trace(go.Scatter(
                x=df_cumulative.index,
                y=df_cumulative[subreddit],
                mode='lines',
                stackgroup='one',
                name=subreddit
            ))

        # Update layout
        fig.update_layout(
            title='Cumulative Total Count of Posts per Subreddit Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative Total Count of Posts',
            showlegend=True
        )

        # Show the figure
        fig.show()

    def visualize_subreddit_counts(self, column_order="subreddit"):
        """
        Visualizes the frequency of subreddit mentions in a bar chart.

        This function takes a dictionary (posts_data) of subreddit names and their 
        corresponding mention counts. It generates a bar chart with pastel colors.

        Args:
            posts_data (dict): A dictionary containing subreddit names as keys 
                            and their mention counts as values.
        """
        # Create a Pandas Series for easier sorting
        subreddit_counts = pd.DataFrame(self.data_structure)
        sorted_subreddits = subreddit_counts[column_order].value_counts().sort_values(ascending=True)
    
        # Generate pastel colors
        start_color = Color("lightblue")
        end_color = Color("lightpink")
        colors = list(start_color.range_to(end_color, len(sorted_subreddits)))
        hex_colors = [color.hex_l for color in colors]

        # Create Plotly bar chart
        fig = go.Figure(
            data=[go.Bar(
                x=sorted_subreddits.index,  # Subreddit names
                y=sorted_subreddits.values,  # Mention counts
                marker_color=hex_colors,     # Pastel colors
            )]
        )

        # Customize the plot layout
        fig.update_layout(
            title="Subreddits Mentioning Search Terms",
            xaxis_title="Subreddit",
            yaxis_title="Number of Mentions",
        )

        fig.show()

    def _apply_condition(self, value, condition):
        """
        Applies a condition to a value using logical operators.

        Args:
        value (float or datetime): The value to be compared.
        condition (str): The condition string (e.g., '<0.5', '>2024-07-01').

        Returns:
        bool: True if the condition is met, False otherwise.
        """
        # Define regex pattern to match logical operators and values
        pattern = r'([<>]=?|==)(.*)'

        # Match the pattern
        match = re.match(pattern, condition)
        if not match:
            raise ValueError("Condition must start with '<', '>', '<=', '>=', or '=='")
        
        operator, condition_value = match.groups()

        # Convert condition_value to appropriate type
        if isinstance(value, datetime):
            value = value.timestamp()
            condition_value = datetime.strptime(condition_value.strip(), '%Y-%m-%d').timestamp()
        else:
            condition_value = float(condition_value.strip())

        # Apply the condition
        if operator == '<=':
            return value <= condition_value
        elif operator == '>=':
            return value >= condition_value
        elif operator == '<':
            return value < condition_value
        elif operator == '>':
            return value > condition_value
        elif operator == '==':
            return value == condition_value
        else:
            raise ValueError("Unsupported operator")

    def filter_posts(self, id_name, post_sentiment=None, created_utc=None, upvote_score=None, upvote_ratio=None, comments_num=None, comments_sentiment_std_dev=None, comments_sentiment_mean=None):
        """
        Filters the collected post data based on specified criteria and returns a new instance of RedditPostsCollector.

        This method applies various filters to the post data stored in the `self.data_structure`
        attribute. Each filter operates on a specific field of the post data, and the filters 
        can be combined to narrow down the dataset based on multiple conditions. The filtered
        data is then used to create and return a new instance of `RedditPostsCollector`, effectively 
        representing a subset of the original data.

        Args:
            id_name (str): A unique identifier for the new filtered instance of `RedditPostsCollector`.
            post_sentiment (str, optional): A comparison string for filtering by post sentiment 
                (e.g., '<0.5', '>0.2'). Defaults to None.
            created_utc (str, optional): A comparison string for filtering by post creation date
                (e.g., '<2024-07-01', '>2024-07-01'). Defaults to None.
            upvote_score (str, optional): A comparison string for filtering by upvote score
                (e.g., '<10', '>5'). Defaults to None.
            upvote_ratio (str, optional): A comparison string for filtering by upvote ratio
                (e.g., '<0.9', '>0.8'). Defaults to None.
            comments_num (str, optional): A comparison string for filtering by the number of comments
                (e.g., '<20', '>5'). Defaults to None.
            comments_sentiment_std_dev (str, optional): A comparison string for filtering by the 
                standard deviation of comment sentiments (e.g., '<0.5', '>0.1'). Defaults to None.
            comments_sentiment_mean (str, optional): A comparison string for filtering by the mean 
                of comment sentiments (e.g., '<0.5', '>0.1'). Defaults to None.

        Returns:
            RedditPostsCollector: A new instance of `RedditPostsCollector` containing the filtered post data.
        """
        filtered_posts = []

        for post in self.data_structure:
            # Filter by post sentiment
            if post_sentiment is not None:
                if not self._apply_condition(post['post_sentiment'], post_sentiment):
                    continue

            # Filter by created_utc
            if created_utc is not None:
                try:
                    post_created_utc = pd.to_datetime(post['post_created_utc'])  # Convert to datetime
                except (TypeError, ValueError):
                    continue  # Skip posts with invalid date formats

                if not self._apply_condition(post_created_utc, created_utc):
                    continue

            # Filter by upvote score
            if upvote_score is not None:
                if not self._apply_condition(post['upvote_score'], upvote_score):
                    continue

            # Filter by upvote ratio
            if upvote_ratio is not None:
                if not self._apply_condition(post['upvote_ratio'], upvote_ratio):
                    continue

            # Filter by number of comments
            if comments_num is not None:
                if not self._apply_condition(post['comments_num'], upvote_ratio):
                    continue

            # Filter by standard deviation of comments sentiment
            if comments_sentiment_std_dev is not None:
                if not self._apply_condition(post['comments_sentiment_std_dev'], upvote_ratio):
                    continue

            # Filter by mean of comments sentiment
            if comments_sentiment_mean is not None:
                if not self._apply_condition(post['comments_sentiment_mean'], upvote_ratio):
                    continue
            # If all filters pass, add the post to the filtered list
            filtered_posts.append(post)

        # TODO 
        self.subreddit_datacollector.add_post_counts(id=self.id)

        return RedditPostsCollector(self.subreddit_datacollector, id=id_name, data_structure=filtered_posts)



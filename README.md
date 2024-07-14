# Reddit EU Elections Sentiment Analysis
This project analyzes sentiment and topics related to the European Union (EU) elections on Reddit. It utilizes the Reddit API (PRAW), natural language processing (NLP) techniques to:

1. Identify subreddits discussing the EU elections.
2. Extract posts and comments mentioning the elections.
3. Analyze sentiment in post titles and comments.
4. Perform topic modeling to discover key themes.
5. Visualize findings using interactive charts and graphs (using Plotly and pyLDAvis).



## Features
* Subreddit Discovery: Automatically identifies subreddits that discuss EU elections based on their descriptions and post content.
* Sentiment Analysis: Analyzes sentiment in both post titles and comments using TextBlob (or other sentiment analysis libraries).
* Topic Modeling: Employs Latent Dirichlet Allocation (LDA) to uncover latent topics within Reddit discussions.

* Interactive Visualizations:
    - Line Charts: Shows the number of posts per day, allowing filtering by date range and subreddit.
    - Scatter Plots: Visualizes the distribution of sentiment polarity for each subreddit over time.
    - Word Clouds: Displays the most relevant words for each topic identified by LDA.
    - pyLDAvis: An interactive visualization tool for exploring topic models.
    - Topic Graphs


# Reddit Analysis Tool

## Libraries
Install the following libraries using pip:

- `praw` (Python Reddit API Wrapper)
- `plotly`
- `dash`
- `textblob` (for sentiment analysis)
- `nltk` (Natural Language Toolkit)
- `pandas`
- `gensim` (for topic modeling)
- `pyLDAvis`
- `scikit-learn` (for machine learning models, if used)
- `langdetect`

## Reddit API Credentials
Create a Reddit app and obtain your `client_id`, `client_secret`, and `user_agent`.
Add your credentials to the code (see the `praw.Reddit()` initialization).

## Setup

### Clone the Repository
```bash
git clone https://github.com/ailine-luconi/Reddit-dynamics-in-eu-elections.git

```


## Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Create a Reddit App
 Obtain your client ID and client secret from the Reddit Developer Portal [https://www.reddit.com/prefs/apps/].

### Configure Credentials:

Either create your own main Python script (``main.py``) or use the Jupyter Notebook ``reddit-research.ipynb``

Replace ``YOUR_CLIENT_ID``, ``YOUR_CLIENT_SECRET``, and ``YOUR_USERNAME`` with your actual Reddit API credentials.

###   Customize search terms in search_terms (if needed).

Run the Script:

```
python main.py
```
Or use the Jupyter Interface to run ``reddit-research.ipynb``

### Explore the Visualizations:
* Cumulative Stacked Area Chart: This chart displays the cumulative total count of posts for each subreddit over time, providing insights into the relative activity and growth patterns of different subreddits.
* pyLDAvis: An interactive topic model visualization should appear in your browser.
* Topic Graph

## Disclaimer:

This code is intended for educational purposes.
Please be mindful of Reddit's API usage guidelines.
Feel free to adapt and extend this code to your specific needs. Contributions and improvements are welcome!
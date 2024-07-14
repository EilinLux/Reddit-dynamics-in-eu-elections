from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import Phrases
import re
import nltk
from pyvis.network import Network
import networkx as nx
import plotly.graph_objects as go
from gensim.models import Word2Vec
from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


def cleaning_text(text, additional_stopwords=None, tokenize=True):
    """
       * Removes URLs: Hyperlinks often contain irrelevant information.
       * Removes Punctuation: Punctuation usually doesn't carry semantic meaning in this context.
       * Removes Non-Alphanumeric Characters: This ensures that only words and numbers remain.
       * Removes Underscores for Hashtags: This splits hashtags into individual words.
    """
    # Low-case
    text= text.lower()

    # Remove URLs and punctuation (including underscores for hashtags)
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+|[^A-Za-z0-9]+", " ", text)
    nltk.download('stopwords') 
    stop_words = set(stopwords.words('english'))
    if additional_stopwords:
        low_case_additional_stopwords = {item.lower() for item in additional_stopwords}
        stop_words.update(low_case_additional_stopwords)
        
    if tokenize:
        #Tokenize
        words = word_tokenize(text)

        filtered_words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 2]


     
def preprocess_text(text, additional_stopwords=None, tokenize=True,stemming_or_lemming_or_none = "lemmatization"  ):
    """
    Preprocesses text for word embedding analysis (e.g., Word2Vec).

    This function performs the following steps:
    1. Low-case: all-low case

    2. Cleaning:
       * Removes URLs: Hyperlinks often contain irrelevant information.
       * Removes Punctuation: Punctuation usually doesn't carry semantic meaning in this context.
       * Removes Non-Alphanumeric Characters: This ensures that only words and numbers remain.
       * Removes Underscores for Hashtags: This splits hashtags into individual words.

    3. Tokenization: Splits the text into individual words.

    4. Stopword Removal: Filters out common English words like "the," "and," "in," which
       usually don't contribute much to the meaning of a sentence.

    5. Length Filtering: Keeps only words that are longer than 2 characters. This removes 
       very short words that might be less meaningful or noisy.

    6. Lowercasing: Converts all words to lowercase to make the analysis case-insensitive.

    7. Optional Stemming/Lemmatization: If enabled, this step reduces words to their base
       or root forms. Stemming is a faster but more aggressive approach, while lemmatization
       is more accurate but computationally expensive. You can choose either "stemming" or
       "lemmatization" by setting the `stemming_or_lemming_or_none` variable.

    Args:
        text: The input text string to be preprocessed.

    Returns:
        A list of preprocessed words.
    """

    # Remove URLs and punctuation (including underscores for hashtags)
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+|[^A-Za-z0-9]+", " ", text)

    nltk.download('stopwords') 
    
    stop_words = set(stopwords.words('english'))

    if additional_stopwords:
        low_case_additional_stopwords = {item.lower() for item in additional_stopwords}
        stop_words.update(low_case_additional_stopwords)
        
    if tokenize:
        #Tokenize
        words = word_tokenize(text)

        filtered_words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 2]

        # Stemming o Lemming or None
        if stemming_or_lemming_or_none == "stemming":
            stemmer = PorterStemmer()
            filtered_words = [stemmer.stem(word) for word in filtered_words]

        elif stemming_or_lemming_or_none == "lemmatization":
            lemmatizer = WordNetLemmatizer()
            filtered_words = [lemmatizer.lemmatize(word) for word in filtered_words]

        return filtered_words
    else:
        return text

def prepate_text_for_nlp_modeling(posts_data, additional_stopwords=None,  stemming_or_lemming_or_none = "lemmatization", comment_words_in=True):
    """
    Combines and preprocesses text from posts and comments, 
    including adding bigrams and trigrams for improved word representations.

    This function iterates through a list of posts and their associated comments, 
    extracts the relevant text (titles, post bodies, comment bodies), and applies
    text preprocessing to normalize the data. It then uses the gensim Phrases model
    to identify and add bigrams (two-word phrases) and trigrams (three-word phrases)
    that occur frequently enough in the corpus. This can help capture meaningful 
    semantic relationships that might be missed by considering words individually.

    Args:
        posts_data: A list of dictionaries representing posts. Each dictionary
                    should have keys 'post_title', 'post_text', and optionally 
                    'comments_bodies' (a list of comment bodies).
        additional_stopwords (list, optional): A list of additional stopwords to remove (default: None).
        comment_words_in (bool, optional): If True, include words from comments in the analysis; otherwise, use only titles (default: False).
        use_stemming (str, optional): use "stemming"; or use "lemmatization" (default: None).

    Returns:
        A list of preprocessed sentences, including added bigrams and trigrams.
    """

    # Combine all text from posts and comments
    all_sentences = []
    for post in posts_data:  # Assuming data_analyzer is your data object
        all_sentences.append(preprocess_text(post['post_text']))
        if comment_words_in:
            for comment in post.get("comments_bodies", []):  # Handle cases where comments might be missing
                all_sentences.append(preprocess_text(comment, additional_stopwords=additional_stopwords, stemming_or_lemming_or_none = stemming_or_lemming_or_none))

    # Add bigrams and trigrams (using gensim Phrases)
    bigram = Phrases(all_sentences, min_count=5, threshold=10)  # Find common bigrams
    trigram = Phrases(bigram[all_sentences], threshold=10)       # Find common trigrams
    all_sentences = list(trigram[bigram[all_sentences]])         # Apply the bigram and trigram models

    return all_sentences


def find_second_element(tuples_list, first_element):
    """
    Finds the second element in a list of tuples where the first element matches the given value.

    Args:
        tuples_list: A list of tuples where each tuple is of the form (first_element, second_element).
        first_element: The first element to search for.

    Returns:
        The second element associated with the first element if found, or None if not found.
    """
    for tup in tuples_list:
        if tup[0] == first_element:
            return tup[1]
    return None  # Return None if the first_element is not found




from gensim.corpora import Dictionary
from gensim.models import Phrases
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer
import itertools

def prepare_text_for_lda(posts_data, stemming_or_lemming_or_none="lemmatization", no_below=5, no_above=0.1, 
                        keep_n=5000, additional_stopwords=None, comment_words_in=True):
    """
    Preprocesses text data from Reddit posts for LDA topic modeling.

    This function performs the following steps:

    1. Tokenization: Splits the text into individual words.
    2. Cleaning: Converts text to lowercase, removes non-alphanumeric characters.
    3. Normalization (Optional): Applies either stemming or lemmatization to reduce words to their base form.
    4. Stop Word Removal (Optional): Removes common words that do not carry significant meaning.
    5. Dictionary Creation: Builds a Gensim dictionary mapping word IDs to words.
    6. Dictionary Filtering: Removes very rare and very common words based on the provided parameters.
    7. Corpus Creation: Converts the text into a bag-of-words format for LDA modeling.

    Args:
        posts_data (list): List of post dictionaries containing 'title' and 'comment_bodies' keys.
        stemming_or_lemming_or_none (bool, optional): If True, use stemming; otherwise, use lemmatization (default: True).
        no_below (int, optional): Keep tokens which are contained in at least `no_below` documents (default: 5).
        no_above (float, optional): Keep tokens which are contained in no more than `no_above` documents (fraction of total corpus size, not absolute number) (default: 0.1).
        keep_n (int, optional): Keep only the first `keep_n` most frequent tokens (default: 5000).


    Returns:
        tuple: A tuple containing the following:
            - dictionary (gensim.corpora.Dictionary): Gensim dictionary of words.
            - corpus (list): Bag-of-words representation of the documents.
            - all_texts (list): List of lists containing the processed tokens for each document.
    """

    all_texts = prepate_text_for_nlp_modeling(posts_data, additional_stopwords=additional_stopwords, stemming_or_lemming_or_none=stemming_or_lemming_or_none, comment_words_in=comment_words_in)

    # Flattening the list of lists to get all tokens for dictionary creation
    # all_tokens = list(itertools.chain.from_iterable(all_texts))
    
    # Create dictionary and corpus (ensure correct texts)
    dictionary = Dictionary(all_texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    corpus = [dictionary.doc2bow(text) for text in all_texts]

    return dictionary, corpus, all_texts  # Return the modified texts as well


import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_wordcloud_grid(lda_model, grid_dimensions="2x3"):
    """
    Generates and displays word clouds for each topic in a grid layout.

    Args:
        lda_model: Trained LDA model object (e.g., from Gensim)
        grid_dimensions (str): A string defining the grid layout in the format "rowsxcolumns" (e.g., "2x3" for a 2 rows by 3 columns grid).

    Raises:
        ValueError: If the grid dimensions are invalid for the number of topics.

    Returns:
        None
    """

    num_topics = lda_model.num_topics
    
    try:
        rows, cols = map(int, grid_dimensions.split("x"))
    except ValueError:
        raise ValueError("Invalid grid dimensions format. Use 'rowsxcolumns' (e.g., '2x3').")

    if rows * cols < num_topics:
        raise ValueError(f"Grid dimensions too small for {num_topics} topics. Please increase rows or columns.")

    # Get the total number of topics
    num_topics = lda_model.num_topics

    # Calculate the size of each sub-plot based on grid dimensions
    fig_width = 10 * cols  
    fig_height = 5 * rows  

    plt.figure(figsize=(fig_width, fig_height))

    for topic_idx, topic in enumerate(lda_model.print_topics(num_topics=num_topics, num_words=10)):  
        topic_words_freq = {}
        for word_weight_pair in topic[1].split("+"):
            weight, word = word_weight_pair.split("*")
            word = word.strip().replace('"', '')
            topic_words_freq[word] = float(weight)

        # Create the word cloud
        wordcloud = WordCloud(background_color="white").generate_from_frequencies(topic_words_freq)

        # Plot the word cloud on the grid
        plt.subplot(rows, cols, topic_idx + 1)  # Create subplots according to the grid dimensions
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Topic #{topic_idx + 1}")

    plt.tight_layout()  # Adjust layout to prevent overlapping titles
    plt.show()



def plot_coherence_values(coherence_values,  topics_start=2, topics_end=21, step=10):
    """
    Plots coherence values using Plotly.

    Args:
    coherence_values (list): List of coherence values.
    limit (int): The upper limit for the x-axis (default: 21).
    start (int): The starting value for the x-axis (default: 2).
    step (int): The step size for the x-axis (default: 2).

    Returns:
    None
    """
    x = list(range(topics_start, topics_end, step))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=coherence_values,
        mode='lines+markers',
        name='Coherence Values'
    ))

    fig.update_layout(
        title='Coherence Score vs. Number of Topics',
        xaxis_title='Num Topics',
        yaxis_title='Coherence Score',
        legend_title='Legend',
        showlegend=True
    )

    fig.show()

def tune_lda_models(corpus, dictionary, texts, topics_start=2, topics_end=21, step=10):
    """
    Tunes LDA models by iteratively training models with different numbers of topics
    and evaluating their coherence.

    This function is designed to help you find the optimal number of topics for your
    LDA model based on coherence scores. It does the following:

    1. Iterates through a range of topic numbers:
       * Starting from `topics_start` (default 2) and ending before `topics_end` (default 21).
       * Increments the number of topics by `step` (default 10) with each iteration.
       * This allows you to explore a wider range of topic numbers efficiently.

    2. Trains LDA models:
       * For each topic number, it trains an LDA model using the provided `corpus` (document-term matrix)
         and `dictionary` (mapping between word IDs and words).
       * It uses a fixed random seed for reproducibility.

    3. Calculates coherence scores:
       * It evaluates the coherence of each trained model using the C_V coherence measure.
       * C_V coherence measures how semantically similar the top words of a topic are.
       * Higher coherence scores generally indicate more interpretable topics.

    4. Stores results:
       * Appends the trained LDA model to the `model_list`.
       * Appends the calculated coherence score to the `coherence_values` list.
       * This allows you to keep track of both the models and their corresponding scores.

    5. Plots coherence values:
       * Calls a separate function `plot_coherence_values` (not shown here) to visualize the
         relationship between the number of topics and the coherence scores.
       * This visualization helps you identify the optimal number of topics.

    Args:
        topics_start (int): The starting number of topics to test (default 2).
        topics_end (int): The ending number of topics to test (exclusive, default 21).
        step (int): The increment by which to increase the number of topics (default 10).

    Returns:
        list: A list of trained LDA models, each with a different number of topics.
    """

    coherence_values = []
    model_list = []
    for num_topics in range(topics_start, topics_end, step):
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=0)
        model_list.append(lda_model)
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())

    plot_coherence_values(coherence_values) 
    return model_list



def visualize_topic_subreddit_graph(subreddit_list, posts_data, dictionary, 
                                    lda_model, num_top_words=10, num_subreddits_limit=None, 
                                    subreddits=None, folder="graphs", filename="topic_subreddit_graph.html"):
   
    """
    Creates and visualizes an interactive network graph of topics, top words, and subreddits.

    Args:
        subreddit_list (list): list of subreddits.
        posts_data (list): List of post dictionaries.
        dictionary (gensim.corpora.Dictionary): Gensim dictionary.
        lda_model (gensim.models.LdaModel): Trained LDA model.
        num_top_words (int): Number of top words to display per topic (default: 10).
        num_subreddits_limit (int, optional): Limit the number of subreddits to include (default: None).
        subreddits (list, optional): List of specific subreddit names to include (default: None).
        folder (str): Folder to save the generated HTML file (default: "graphs").
        filename (str): Filename for the generated HTML file (default: "topic_subreddit_graph.html").

    Returns:
        str: Path to the generated HTML file.
    """
   
    try:
        # Calculate topic assignments for each subreddit based on post content using the LDA model.
        topic_assignments = {}
        texts = []  # Create a texts variable
        for idx, post in enumerate(posts_data):
            texts.append(nltk.word_tokenize(post['title'].lower()) + nltk.word_tokenize(" ".join(post['comment_bodies']).lower()))
            bow_vector = dictionary.doc2bow(texts[idx])
            topics = lda_model.get_document_topics(bow_vector)

            subreddit = post["subreddit"]
            if subreddit not in topic_assignments:
                topic_assignments[subreddit] = {}

            for topic_id, topic_prob in topics:
                if topic_id not in topic_assignments[subreddit]:
                    topic_assignments[subreddit][topic_id] = 0
                topic_assignments[subreddit][topic_id] += topic_prob

        # Filter subreddits if limits or specific list provided
        if num_subreddits_limit is not None:
            topic_assignments = dict(list(topic_assignments.items())[:num_subreddits_limit])
        elif subreddits is not None:
            topic_assignments = {subreddit: topic_assignments[subreddit] for subreddit in subreddits if subreddit in topic_assignments}

        # Create the graph
        G = nx.Graph()

        # Add topic nodes
        for topic_idx in range(lda_model.num_topics):
            G.add_node(f"Topic {topic_idx + 1}", color="skyblue")

        # Add word nodes for each topic
        for topic_idx, topic_words in lda_model.print_topics(num_topics=lda_model.num_topics, num_words=num_top_words):
            for word_weight_pair in topic_words.split("+"):
                weight, word = word_weight_pair.split("*")
                word = word.strip().replace('"', '')
                G.add_node(word, color="lightgreen")
                G.add_edge(f"Topic {topic_idx + 1}", word)

        # Add subreddit nodes
        for subreddit_name in subreddit_list:
            G.add_node(subreddit_name, color="green")

        # Add edges between topics and subreddits
        for subreddit_name, topic_assignments in topic_assignments.items():
            for topic_idx, weight in topic_assignments.items():
                G.add_edge(subreddit_name, f"Topic {topic_idx + 1}", weight=weight)

        # Create the Pyvis network
        net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")
        net.from_nx(G)

        # Customize node appearance
        for node in net.nodes:
            if node['label'].startswith("Topic"):
                node['color'] = 'green'
                node['size'] = 10 + 20 * sum(topic_assignments.get(node['label'], {}).values())
            elif node['label'] in subreddit_list:
                node['color'] = 'coral'
            else:  # Word nodes
                node['color'] = 'blue'
                node['size'] = 10

        # Show the graph
        # display(HTML(net.generate_html()))
            # Add a timestamp to the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{folder}/{timestamp}_{filename}"
        return net.show(filename)

    except Exception as e:
        print(f"An error occurred during graph visualization: {e}")


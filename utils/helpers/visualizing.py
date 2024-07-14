import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import plotly.express as px
import pandas as pd
import math 
import plotly.subplots as sp
from datetime import datetime

def visualize_percentage_of_posts(posts_data_list, layout="1x2", columns_names=["2019", "2024"], top_n_subreddits=10):
    """
    Calculates and visualizes the percentage of posts per subreddit, limiting displayed subreddits but not calculations.

    Args:
        posts_data_list (list): A list of lists, where each inner list contains dictionaries representing posts.
        layout (str, optional): A string specifying the grid layout (e.g., "2x2", "3x1"). Defaults to "1x2".
        columns_names (list, optional): A list of names for the datasets, used in pie chart titles. Defaults to ["2019", "2024"].
        top_n_subreddits (int, optional): The number of top subreddits to display in each pie chart. Defaults to 10.

    Returns:
        None (Displays the Plotly figure)
    """

    num_datasets = len(posts_data_list)
    try:
        rows, cols = map(int, layout.split("x"))
    except ValueError:
        raise ValueError("Invalid layout format. Use 'rowsxcols' (e.g., '2x2').")

    if rows * cols < num_datasets:
        raise ValueError(f"Not enough subplots for {num_datasets} datasets. Increase rows or columns in layout.")

    fig = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'domain'} for _ in range(cols)] for _ in range(rows)])

    for i, posts_data in enumerate(posts_data_list):
        # Count posts per subreddit
        subreddit_counts = {}
        for post in posts_data:
            subreddit = post['subreddit']
            subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1

        # Calculate percentages based on ALL subreddits
        total_posts = len(posts_data)
        subreddit_percentages_all = {subreddit: (count / total_posts) * 100 for subreddit, count in subreddit_counts.items()}

        # Get the top N subreddits for display
        top_subreddits = sorted(subreddit_counts, key=subreddit_counts.get, reverse=True)[:top_n_subreddits]

        # Filter percentages for display
        subreddit_percentages = {subreddit: subreddit_percentages_all[subreddit] for subreddit in top_subreddits}

        # Calculate "Other" category percentage
        other_percentage = 100 - sum(subreddit_percentages.values())
        subreddit_percentages["Other"] = other_percentage  # Add "Other" to pie chart

        # Add pie chart to subplot
        row = i // cols + 1
        col = i % cols + 1
        fig.add_trace(go.Pie(labels=list(subreddit_percentages.keys()), 
                             values=list(subreddit_percentages.values()), 
                             name=f"Data {columns_names[i]}"), 
                             row=row, col=col)

    # Update layout
    fig.update_layout(title_text='Percentage of Posts per Subreddit (Top N Displayed)')

    # Show the figure
    fig.show()



def visualize_scatter_plots(posts_data_list, subreddit_filter=None, x="post_upvote_score", y="post_upvote_ratio", layout="1x1", scale=True, columns_names=["2019", "2024"]):
    """
    Creates interactive scatter plots for multiple datasets, with filtering, color-coding, and optional shared scaling.

    Args:
        posts_data_list (list): List of lists, where each inner list contains dictionaries of post data.
        subreddit_filter (list, optional): Subreddits to filter by. If None, no filter is applied.
        x (str): The name of the column to be used for the x-axis.
        y (str): The name of the column to be used for the y-axis.
        layout (str): Specifies the subplot layout, e.g., "2x2" for a 2 by 2 grid.
        columns_names (list): List of names for the datasets.
        scale (bool, optional): If True, subplots share the same scale. Defaults to True.

    Returns:
        None (Displays an interactive Plotly figure with subplots)
    """

    num_datasets = len([data for data in posts_data_list if data])  # Count non-empty datasets

    try:
        num_rows, num_cols = map(int, layout.split("x"))
        if num_rows * num_cols < num_datasets:  # Check if layout can accommodate the datasets
            num_rows = math.ceil(num_datasets / 2)  # Adjust to fit all plots
            num_cols = 2
            print(f"Adjusted layout to {num_rows}x{num_cols} to fit all datasets.")
    except ValueError:
        print("Invalid layout format. Using default 1x1 layout.")
        num_rows, num_cols = 1, 1

    fig = sp.make_subplots(
        rows=num_rows, cols=num_cols, subplot_titles=[f"Dataset {i}" for i in columns_names]
    )

    num_rows, num_cols = map(int, layout.split("x"))
    # Create subplots, sharing axes if scale=True
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[f"Dataset {i}" for i in columns_names],
                        shared_xaxes=scale, shared_yaxes=scale)

    x_label = x.replace("_", " ").capitalize()
    y_label = y.replace("_", " ").capitalize()
    all_subreddits = set()
    subreddit_legend_shown = set()  # To keep track of which subreddits have been shown in the legend
    # Initialize min/max values for x and y axes if scaling is enabled
    if scale:
        x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')

    for i, posts_data in enumerate(posts_data_list):
        if not posts_data:  # Skip empty datasets
            continue

        if subreddit_filter:
            filtered_posts = [post for post in posts_data if post.get('subreddit') in subreddit_filter]
        else:
            filtered_posts = posts_data

        df = pd.DataFrame(filtered_posts)
        all_subreddits.update(df["subreddit"].unique())

        # Update min/max values for x and y axes if scaling is enabled
        if scale:
            x_min = min(x_min, df[x].min())
            x_max = max(x_max, df[x].max())
            y_min = min(y_min, df[y].min())
            y_max = max(y_max, df[y].max())

        # Dynamic color generation (same color for same subreddit across plots)
        subreddit_colors = {sub: "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for sub in all_subreddits}
        df["color"] = df["subreddit"].apply(lambda sub: subreddit_colors.get(sub, "gray"))

        # Create the scatter plot for this dataset
        scatter_fig = px.scatter(
            df,
            x=x,
            y=y,
            color="subreddit",
            hover_name="post_title",
            labels={"x": x_label, "y": y_label},
            category_orders={"color": list(all_subreddits)},
        )

        # Add traces to subplot, hiding the legend for repeated subreddits
        for trace in scatter_fig.data:
            if trace.name in subreddit_legend_shown:
                trace.showlegend = False
            else:
                subreddit_legend_shown.add(trace.name)
            fig.add_trace(trace, row=i // num_cols + 1, col=i % num_cols + 1)
    # Apply shared axes ranges if scaling is enabled
    if scale:
        for i in range(1, num_datasets + 1):  # Update all subplots' axes
            fig.update_xaxes(range=[x_min, x_max], row=i // num_cols + 1, col=i % num_cols + 1)
            fig.update_yaxes(range=[y_min, y_max], row=i // num_cols + 1, col=i % num_cols + 1)

    # Update layout
    fig.update_layout(
        height=400 * num_rows,   
        width=600 * num_cols,    
        xaxis_title=x_label, 
        yaxis_title=y_label,
        template="plotly_white",
        showlegend=True,
        title=f"{x_label} vs. {y_label} {'(Filtered by ' + ', '.join(subreddit_filter) + ')' if subreddit_filter else ''}",
    )

    fig.show()


import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots

def visualize_subreddits_details_scatter_plot(dataframes, x_column, y_column, label_column=None, title='Scatter Plot', scale=True, 
                                               subreddit_filter=None, filter_column=None, filter_values=None, filter_mode='include', layout="1x2"):
    """
    Creates an interactive scatter plot grid using Plotly with labels, color-coded by subreddit, and optional filtering.

    Args:
        dataframes (list): A list of pandas DataFrames to be displayed side-by-side.
        x_column (str): The name of the column to use for the x-axis.
        y_column (str): The name of the column to use for the y-axis.
        label_column (str, optional): The name of the column to use for point labels (hover text). 
                                      Defaults to None, in which case the index values will be used.
        title (str, optional): The title of the overall plot. Defaults to 'Scatter Plot'.
        subreddit_filter (str, optional): The name of the subreddit column or index to filter by. 
                                          Defaults to None (no filtering).
        filter_column (str, optional): The name of the column to filter on within the filtered subreddit(s). 
                                      Defaults to None (no filtering).
        filter_values (list, optional): The values to filter for in the filter_column. Defaults to None (no filtering).
        filter_mode (str, optional): Either 'include' to include only the specified values 
                                      or 'exclude' to exclude them. Defaults to 'include'.
        layout (str, optional): The layout of subplots in the format "rowsxcols" (e.g., "1x2", "2x2"). Defaults to "1x2".
        scale (bool, optional): If True, the subplots will share the same scale for both axes. Defaults to True.


    Returns:
        None: Displays the interactive Plotly chart in your web browser.
    """
    num_datasets = len(dataframes)
    try:
        rows, cols = map(int, layout.split("x"))
    except ValueError:
        raise ValueError("Invalid layout format. Use 'rowsxcols' (e.g., '2x2').")

    if rows * cols < num_datasets:
        raise ValueError(f"Not enough subplots for {num_datasets} datasets. Increase rows or columns in layout.")

    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=scale, shared_yaxes=scale)  # Enable shared axes based on scale

    if scale == True: 
        # Calculate min and max values for x and y axes across all dataframes
        x_min = min([df[x_column].min() for df in dataframes])
        x_max = max([df[x_column].max() for df in dataframes])
        y_min = min([df[y_column].min() for df in dataframes])
        y_max = max([df[y_column].max() for df in dataframes])


    for i, df in enumerate(dataframes):
        # Handle grouped DataFrames
        if isinstance(df.index, pd.MultiIndex) and not subreddit_filter:
            df = df.reset_index()
            subreddit_filter = df.index.name

        # Apply filtering if specified
        filtered_df = df.copy()
        filtered_df = filtered_df.dropna(subset=[x_column, y_column])  # Filter out null values
        
        if filter_column and filter_values:
            filtered_df = filtered_df.dropna(subset=[label_column])
            if filter_mode == 'include':
                filtered_df = filtered_df[filtered_df[filter_column].isin(filter_values)]
            elif filter_mode == 'exclude':
                filtered_df = filtered_df[~filtered_df[filter_column].isin(filter_values)]
            else:
                raise ValueError("Invalid filter_mode. Choose from 'include' or 'exclude'.")

        # Determine the label column
        if label_column:
            hover_name = label_column
        elif isinstance(filtered_df.index, pd.MultiIndex):
            hover_name = [', '.join(map(str, idx)) for idx in filtered_df.index]
        else:
            hover_name = filtered_df.index 

        # Create the scatter plot for the current DataFrame
        fig.add_trace(
            px.scatter(filtered_df, x=x_column, y=y_column, hover_name=hover_name, color=subreddit_filter, title=title).data[0],
            row=i // cols + 1,  # Row number for this subplot
            col=i % cols + 1   # Column number for this subplot
        )

        # Calculate mean and standard deviation for this DataFrame
        y_mean = filtered_df[y_column].mean()
        y_std = filtered_df[y_column].std()
        

        # Add lines for one standard deviation above and below the mean for this subplot
        fig.add_hline(
            y=y_mean, line_dash="dash", line_color="red", annotation_text="Mean", annotation_position="bottom right",
            row=i // cols + 1, col=i % cols + 1
        )
        fig.add_hline(
            y=y_mean + y_std, line_dash="dot", line_color="blue", annotation_text="+1 Std Dev", annotation_position="top right",
            row=i // cols + 1, col=i % cols + 1
        )
        fig.add_hline(
            y=y_mean - y_std, line_dash="dot", line_color="blue", annotation_text="-1 Std Dev", annotation_position="bottom right",
            row=i // cols + 1, col=i % cols + 1
        )
    if scale == True:
            # Set shared x and y axes ranges
            fig.update_xaxes(range=[x_min, x_max])
            fig.update_yaxes(range=[y_min, y_max])
            
    # Update the overall layout
    fig.update_layout(
        showlegend=True,  
        xaxis_title=x_column.replace("_", " "), 
        yaxis_title=y_column.replace("_", " "),
    )

    fig.show()  # Display the figure


import plotly.graph_objects as go
import nltk
import pandas as pd


def create_subreddit_topic_heatmaps(posts_data, top_n_topics=20, list_stop_words_extension=["eu","europe","elections","european","election","european elections"]):
    """
    Creates a heatmap visualizing topic frequency across subreddits,
    colored by the sentiment of those topics within post titles.

    Args:
        posts_data (list): A list of dictionaries, each containing post data
            with 'subreddit', 'title', and 'post_sentiment' keys.
        top_n_topics (int, optional): The number of top topics to display. Defaults to 20.
    """
        
    list_stop_words = nltk.corpus.stopwords.words('english')
    list_stop_words.extend(list_stop_words_extension)
    # Prepare data for heatmap with sentiment
    subreddit_topic_freq = {}
    subreddit_topic_sentiment = {}

    for post in posts_data:
        subreddit = post['subreddit']
        title_words = nltk.word_tokenize(post['title'].lower())

        for word in title_words:
            if word not in list_stop_words and word.isalnum():
                if subreddit not in subreddit_topic_freq:
                    subreddit_topic_freq[subreddit] = {}
                    subreddit_topic_sentiment[subreddit] = {}

                subreddit_topic_freq[subreddit][word] = subreddit_topic_freq[subreddit].get(word, 0) + 1

                # Use post sentiment directly
                subreddit_topic_sentiment[subreddit][word] = post['post_sentiment']

    # Convert to DataFrames
    df_freq = pd.DataFrame(subreddit_topic_freq).fillna(0).T
    df_sentiment = pd.DataFrame(subreddit_topic_sentiment).fillna(0).T

    # Filter to top N topics
    df_freq = df_freq[df_freq.sum(axis=0).nlargest(top_n_topics).index]
    df_sentiment = df_sentiment[df_freq.sum(axis=0).index]

    # Create Plotly heatmap with sentiment coloring
    fig = go.Figure(data=go.Heatmap(
        z=df_freq.values,
        x=df_freq.columns,
        y=df_freq.index,
        colorscale='Viridis',  # You can choose a different colorscale
        colorbar=dict(title='Frequency'),
        hoverongaps=False  # Show hover information for empty cells
    ))

    # Add sentiment markers
    for i, row in df_sentiment.iterrows():
        for j, sentiment in enumerate(row):
            freq = df_freq.loc[i, df_freq.columns[j]]
            if freq > 0:
                marker_color = 'green' if sentiment > 0 else 'red' if sentiment < 0 else 'grey'
                fig.add_trace(go.Scatter(
                    x=[df_freq.columns[j]],
                    y=[i],
                    mode='markers',
                    marker=dict(size=freq*2, color=marker_color),
                    showlegend=False,
                    hovertemplate='Topic: %{x}<br>Subreddit: %{y}'
                ))

    fig.update_layout(
        title='Topic Frequency Heatmap by Subreddit (Colored by Sentiment in Titles)',
        xaxis_title='Topic',
        yaxis_title='Subreddit',
        yaxis_autorange='reversed'  # Reverse y-axis for readability
    )

    fig.show()


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


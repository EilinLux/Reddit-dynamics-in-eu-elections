def contains_keyword(text, keywords):
    """Checks if any of the keywords are present in the text (case-insensitive)."""
    text_lower = text.lower()
    for keyword in keywords:
        if keyword.lower() in text_lower:
            return True
    return False

def filter_posts_by_keywords(posts_data, keywords):
    """Filters posts based on the presence of keywords in the title, text, or comments."""
    filtered_posts = []
    for post in posts_data:
        if (
            contains_keyword(post['post_title'], keywords) or
            contains_keyword(post['post_text'], keywords) or
            any(contains_keyword(comment, keywords) for comment in post.get("comments_bodies", []))
        ):
            filtered_posts.append(post)
    return filtered_posts
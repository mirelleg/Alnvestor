import praw

# Connect to Reddit
reddit = praw.Reddit(
    client_id="hXa0VNF87XwdKlJEcwCuLA",
    client_secret="hJaEUA7bJnYw7FgY-Nr1gWzKuWDFMg",
    user_agent="AInvestor Hackathon project for LAHack"
)

# Stock keyword to search
keyword = "TSLA"

# List of target subreddits
subreddits = [
    "wallstreetbets", "Superstonk", "options", "stocks", "StockMarket"
]

# Container for collected posts
posts = []

# Search posts from each subreddit
for sub_name in subreddits:
    try:
        subreddit = reddit.subreddit(sub_name)

        for post in subreddit.search(query=keyword, sort="new", time_filter="month", limit=100):
            # Filter: text length > 50 characters and score > 10
            if len(post.selftext) > 50 and post.score > 10:
                posts.append({
                    "title": post.title,
                    "text": post.selftext,
                    "score": post.score,
                    "created_utc": post.created_utc,
                    "url": post.url
                })
        
            
            # Stop early if already have 100 posts
            if len(posts) >= 100:
                break
    
    except Exception as e:
        print(f"Error accessing subreddit {sub_name}: {e}")
    
    if len(posts) >= 100:
        break

# Sort posts by creation time (newest first)
posts.sort(key=lambda x: x["created_utc"], reverse=True)

# Create a list of "Title Text" combined strings
post_pairs = [f"{p['title']} {p['text']}" for p in posts]
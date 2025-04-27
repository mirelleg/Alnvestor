import praw

# Function to fetch Reddit posts related to a stock keyword
def fetch_reddit_posts(keyword):
    reddit = praw.Reddit(
        client_id="hXa0VNF87XwdKlJEcwCuLA",
        client_secret="hJaEUA7bJnYw7FgY-Nr1gWzKuWDFMg",
        user_agent="AInvestor Hackathon project for LAHack"
    )

    subreddits = [
        "wallstreetbets", "Superstonk", "options", "stocks", "StockMarket"
    ]

    posts = []

    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.search(query=keyword, sort="new", time_filter="month", limit=100):
                if len(post.selftext) > 50 and post.score > 10:
                    posts.append(f"{post.title}$$${post.selftext}")

                if len(posts) >= 100:
                    break
        except Exception as e:
            print(f"Error accessing subreddit {sub_name}: {e}")

        if len(posts) >= 100:
            break

    return posts

if __name__ == "__main__":
    ticker = input("Please enter a stock ticker (e.g., TSLA): ").strip().upper()
    posts = fetch_reddit_posts(ticker)

    if posts:
        for post in posts:
            print(post)
            print("-" * 60)
    else:
        print("No relevant Reddit posts found for this ticker.")
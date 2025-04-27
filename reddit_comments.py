import praw
import datetime

def fetch_recent_reddit_posts(keyword, max_posts=3):
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
            for post in subreddit.search(query=keyword, sort="new", time_filter="month", limit=30):
                if len(post.selftext) > 50 and post.score > 10:  # 可以稍微降低阈值，防止太少
                    posts.append({
                        "title": post.title,
                        "text": post.selftext,
                        "created_time": datetime.datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                    })
                if len(posts) >= max_posts:
                    break
        except Exception as e:
            print(f"Error accessing subreddit {sub_name}: {e}")

        if len(posts) >= max_posts:
            break

    return posts

if __name__ == "__main__":
    ticker = input("Please enter a stock ticker (e.g., TSLA): ").strip().upper()
    recent_posts = fetch_recent_reddit_posts(ticker)

    if recent_posts:
        for idx, post in enumerate(recent_posts, 1):
            print(f"Title: {post['title']}")
            print(f"Time: {post['created_time']} UTC")
            print(f"Content: {post['text']}\n")
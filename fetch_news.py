import feedparser

def fetch_headlines():
    url = "http://feeds.bbci.co.uk/news/rss.xml"  # BBC News RSS Feed
    feed = feedparser.parse(url)
    

    headlines = [entry.title + " " + entry.get("description", "") for entry in feed.entries]

    return headlines

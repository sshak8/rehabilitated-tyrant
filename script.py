import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import feedparser
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import plotly.express as px
import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud

# Ensure NLTK data is downloaded quietly on startup
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


def remove_html_tags(text):
    return re.sub(r"<[^>]+>", "", text)


@st.cache_data(ttl=600)
def fetch_news(topic, max_articles=100):
    encoded_topic = quote(topic)
    url = f"https://news.google.com/rss/search?q={encoded_topic}"

    # Custom User-Agent prevents Google RSS from blocking feedparser
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    try:
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            feed = feedparser.parse(xml_data)
    except Exception as e:
        st.error(f"Error fetching news feed: {e}")
        return []

    articles = []
    for entry in feed.entries[:max_articles]:
        title = entry.title if "title" in entry else ""
        summary = remove_html_tags(entry.summary) if "summary" in entry else ""
        link = entry.link if "link" in entry else ""
        published = entry.published if "published" in entry else ""

        if title:
            articles.append(
                {
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "published": published,
                }
            )

    return articles


def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return {
        "sentiment": sentiment,
        "polarity": polarity,
        "subjectivity": analysis.sentiment.subjectivity,
    }


def process_single_article(article):
    full_text = article["title"] + " " + article["summary"]
    sentiment_data = analyze_sentiment(full_text)
    article.update(sentiment_data)
    return article


def process_articles(articles):
    total_articles = len(articles)
    if total_articles == 0:
        return []

    workers = min(8, total_articles)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        processed_articles = list(executor.map(process_single_article, articles))

    return processed_articles


def plot_sentiment_distribution(df):
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    fig = px.pie(
        sentiment_counts,
        values="Count",
        names="Sentiment",
        title="Sentiment Distribution",
    )
    return fig


def plot_polarity_subjectivity(df):
    fig = px.scatter(
        df,
        x="polarity",
        y="subjectivity",
        color="sentiment",
        hover_data=["title"],
    )
    fig.update_layout(
        xaxis_title="Polarity (Negative ↔ Positive)",
        yaxis_title="Subjectivity (Factual ↔ Opinion)",
    )
    return fig


def generate_word_cloud(text):
    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig


def main():
    st.set_page_config(
        page_title="News Sentiment Analyzer",
        layout="wide",
    )
    st.title("News Sentiment Analyzer")

    if "sentiment_df" not in st.session_state:
        st.session_state.sentiment_df = None

    with st.sidebar:
        st.title("Settings")
        topic = st.text_input("Search Topic:", value="technology")
        max_articles = st.slider(
            "Max Articles:", min_value=10, max_value=1000, value=100
        )
        st.markdown("---")
        st.subheader("About")
        st.info(
            "This tool analyzes the sentiment of recent news articles with the use of AI"
        )

        if st.button("Analyze news for me"):
            with st.spinner("Please hold for Analysis!..."):
                articles = fetch_news(topic, max_articles)
                if articles:
                    processed = process_articles(articles)
                    st.session_state.sentiment_df = pd.DataFrame(processed)
                    st.success(f"Analyzed {len(processed)} news articles!")
                else:
                    st.warning("No articles found for this topic.")

    # Display Dashboard Results
    if st.session_state.sentiment_df is not None and not st.session_state.sentiment_df.empty:
        df = st.session_state.sentiment_df

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_sentiment_distribution(df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_polarity_subjectivity(df), use_container_width=True)

        st.subheader("Jargon Cloud")
        all_text = " ".join(df["title"] + " " + df["summary"])
        if all_text.strip():
            fig_wc = generate_word_cloud(all_text)
            st.pyplot(fig_wc)

        st.subheader("Raw Data")
        st.dataframe(df)
    else:
        st.info("Enter a topic in the sidebar and click **Analyze news for me** to start.")


if __name__ == "__main__":
    main()

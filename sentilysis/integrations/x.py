import time
from playwright.sync_api import sync_playwright
from stalkers.rl_agent.test import get_action


def scrape_tweets(username, deep_search_factor=5):
    tweets = list()
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)  # Use headless=False for debugging
        page = browser.new_page()
        page.goto(f"https://x.com/{username}")
        prev_page = page.content()
        for _ in range(deep_search_factor):
            action = get_action(page.content(), prev_page)
            if action < 1:
                page.mouse.wheel(0, 10000)
            elif action < 2:
                x, y = (
                    get_action(page.content(), prev_page),
                    get_action(page.content(), prev_page),
                )
                page.mouse.click(x, y)
            elif action < 3:
                page.screenshot(path="screenshot.png")
            elif action < 4:
                ...
            time.sleep(0.5)

            data = get_data(page)
            tweets.extend(data)

        browser.close()
        return tweets


def get_data(page):
    articles = page.locator("article")
    tweets = []
    for i in range(articles.count()):
        try:
            tweet_text = (
                articles.nth(i).locator('[data-testid="tweetText"]').inner_text()
            )
            time_element = articles.nth(i).locator("time")
            timestamp = (
                time_element.get_attribute("datetime")
                if time_element.count() > 0
                else None
            )
            tweets.append({"text": tweet_text, "timestamp": timestamp})
        except:
            continue
    return tweets

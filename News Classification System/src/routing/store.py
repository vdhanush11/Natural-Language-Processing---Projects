# Lifecycle stage 9 — Category-feed routing
 
import csv
import os
from datetime import datetime
 
FEED_DIR = "data/feeds"
 
 
def route_article(text, category):
 
    os.makedirs(FEED_DIR, exist_ok=True)
 
    path = f"{FEED_DIR}/{category}.csv"
 
    new_file = not os.path.exists(path)
 
    with open(path, "a", newline="", encoding="utf-8") as f:
 
        writer = csv.writer(f)
 
        if new_file:
            writer.writerow(["timestamp", "category", "text"])
 
        writer.writerow([datetime.now().isoformat(), category, text])

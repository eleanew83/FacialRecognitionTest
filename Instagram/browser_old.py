import browser_cookie3
import instaloader
import os
from datetime import datetime

# Initialize Instaloader
L = instaloader.Instaloader()

# Load Instagram cookies from Chrome
cookies = browser_cookie3.chrome(domain_name="instagram.com")

# Use these cookies for Instaloader
L.context._session.cookies.update(cookies)
print("✅ Loaded Instagram session from browser cookies!")

# Test login
profile = instaloader.Profile.from_username(L.context, "gibraltarmacaques")
print(f"Logged in as {profile.username}")


# Set a custom User-Agent (mobile version of Instagram)
L.context._session.headers.update({
    'User-Agent': 'Instagram 123.1.0.26.115 Android'
})


# Define the hashtag to search
HASHTAG = "gibraltarmacaques"  # Change this to your desired hashtag

# Define the date range (YYYY, MM, DD)
START_DATE = datetime(2025, 1, 1)  # Change this to your start date
END_DATE = datetime(2025, 2, 1)    # Change this to your end date

# Get the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the folder inside the script's directory
download_folder = os.path.join(script_dir, HASHTAG)

# Ensure the folder exists
os.makedirs(download_folder, exist_ok=True)

# Fetch posts with error handling
try:
    hashtag = instaloader.Hashtag.from_name(L.context, HASHTAG)
    for post in hashtag.get_posts():
        post_date = post.date

        # Check if post is within the date range
        if START_DATE <= post_date <= END_DATE:
            print(f"📥 Downloading {post.shortcode} from {post_date}")
            L.download_post(post, target=download_folder)

except instaloader.exceptions.ConnectionException as e:
    print(f"❌ Error fetching posts: {e}")
except instaloader.exceptions.LoginRequiredException:
    print("❌ Instagram requires login verification. Please verify in your browser and retry.")
except instaloader.exceptions.TooManyRequestsException:
    print("⚠️ Too many requests! Instagram might be rate-limiting you. Try again later.")
except Exception as e:
    print(f"⚠️ An unexpected error occurred: {e}")


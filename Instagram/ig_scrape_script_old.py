import instaloader
import os
from datetime import datetime

# Initialize Instaloader
L = instaloader.Instaloader()

# Instagram login credentials
USERNAME = "gibraltarmacaques"  # Change to your IG username
SESSION_FILE = f"{USERNAME}.session"  # Session file name

# Attempt to load session file
try:
    L.load_session_from_file(USERNAME)
    print("✅ Session loaded successfully!")
except FileNotFoundError:
    print("⚠️ Session file not found. Logging in manually...")
    try:
        L.login(USERNAME, input("Enter your Instagram password: "))  # Secure password input
        L.save_session_to_file()  # Save session for future use
        print("✅ Session saved!")
    except instaloader.exceptions.ConnectionException as e:
        print(f"❌ Login failed: {e}")
        exit()

# Define the hashtag to search
HASHTAG = "gibraltarmacaques"  # Change this to your desired hashtag

# Define the date range (YYYY, MM, DD)
START_DATE = datetime(2026, 2, 1)  # Change this to your start date
END_DATE = datetime(2026, 2, 24)    # Change this to your end date
MAX_POSTS = 30 # Set max posts to download

# Get the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the folder inside the script's directory
download_folder = os.path.join(script_dir, HASHTAG)

# Ensure the folder exists
os.makedirs(download_folder, exist_ok=True)

# Fetch posts with error handling
count = 0
try:
    hashtag = instaloader.Hashtag.from_name(L.context, HASHTAG)
    for post in hashtag.get_posts():
        if count >= MAX_POSTS:
            break

        post_date = post.date
        # Check if post is within the date range
        if START_DATE <= post_date <= END_DATE:
            print(f"📥 Downloading {post.shortcode} from {post_date}")
            L.download_post(post, target=download_folder)
            count += 1
except instaloader.exceptions.ConnectionException as e:
    print(f"❌ Error fetching posts: {e}")
except instaloader.exceptions.LoginRequiredException:
    print("❌ Instagram requires login verification. Please verify in your browser and retry.")
except instaloader.exceptions.TooManyRequestsException:
    print("⚠️ Too many requests! Instagram might be rate-limiting you. Try again later.")
except Exception as e:
    print(f"⚠️ An unexpected error occurred: {e}")

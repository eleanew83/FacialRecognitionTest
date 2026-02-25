import os
import time
import requests
import pandas as pd
import re
import browser_cookie3
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup


def parse_description(description):
    """Extract username, date, and content from Instagram description."""
    
    # 🔹 Extract username (before "on")
    username_match = re.search(r"(\w+) on", description)
    username = username_match.group(1) if username_match else "❌ No Username Found"

    # 🔹 Extract date (between "on" and `:`)
    date_match = re.search(r"on (.*?)\:", description)
    post_date = date_match.group(1) if date_match else "❌ No Date Found"

    # 🔹 Extract content within quotes `""`
    content_match = re.search(r'"([^"]+)"', description, re.DOTALL)
    content = content_match.group(1).strip() if content_match else "❌ No Content Found"

    return username, post_date, content


def get_location(post_soup):
    """Extracts location from Instagram post JSON data."""
    # 🔹 Find <a> tag that has 'x1i10hfl' class and contains 'locations' in href
    location_tag = post_soup.find("a", class_="x1i10hfl", href=lambda href: href and "locations" in href)
    
    # 🔹 Extract location name from anchor text
    if location_tag and location_tag.text.strip().lower() != "locations":
        return location_tag.text.strip()
    else:
        return "❌ No Location Found"



# 🔹 Set the hashtag to scrape
HASHTAG = "visitgibraltar" # "gibraltarmacaques"

# 🔹 Instagram login credentials (if needed)
USERNAME = "gibraltarmacaques"
PASSWORD = "macca00777@"

# 🔹 Folder to save images
save_folder = os.path.join(os.getcwd(), HASHTAG)
os.makedirs(save_folder, exist_ok=True)

# 🔹 Setup Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in headless mode (set to False for debugging)
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")

# Initialize WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# 🔹 Load Instagram session cookies from Chrome (so we appear logged in)
driver.get("https://www.instagram.com")
for cookie in browser_cookie3.chrome(domain_name="instagram.com"):
    try:
        driver.add_cookie({"name": cookie.name, "value": cookie.value, "domain": cookie.domain})
    except Exception:
        pass
print("✅ Loaded Chrome cookies into Selenium")

# # 🔹 Step 1: Log Into Instagram
# print("🔑 Logging into Instagram...")
# driver.get("https://www.instagram.com/accounts/login/")
# time.sleep(5)

# # Find login fields and enter credentials
# username_input = driver.find_element(By.NAME, "username")
# password_input = driver.find_element(By.NAME, "password")

# username_input.send_keys(USERNAME)
# password_input.send_keys(PASSWORD)
# password_input.send_keys(Keys.RETURN)

# time.sleep(5)  # Wait for login to process

# # Verify login success by checking if redirected to the home page
# if "accounts/login" in driver.current_url:
#     print("❌ Login failed. Check credentials or complete CAPTCHA manually.")
#     driver.quit()
#     exit()
# else:
#     print("✅ Successfully logged in!")

# 🔹 Step 1: Open Instagram Hashtag Page
url = f"https://www.instagram.com/explore/tags/{HASHTAG}/"
print(f"🌐 Opening URL: {url}")
driver.get(url)
time.sleep(5)  # Wait for page to load
print("🔍 Current URL:", driver.current_url)

# 🔹 Step 2: Scroll to Load More Posts
for scroll in range(10):  # Adjust scrolling as needed — more scrolls = more posts loaded
    print(f"📜 Scrolling down... ({scroll + 1}/10)")
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
    time.sleep(3)

# Debug: Print HTML again after scrolling
soup = BeautifulSoup(driver.page_source, "html.parser")
print("\n🔍 DEBUG: Page source after scrolling (first 500 chars):")
# print(driver.page_source[:500])
# print(soup)

# 🔹 Step 3: Extract Post URLs
post_links = [a["href"] for a in soup.find_all("a", href=True) if "/p/" in a["href"]]
print(f"✅ Found {len(post_links)} posts!")

# Debug: Print the first few extracted post links
if post_links:
    print("\n🔍 DEBUG: First few post links:")
    for link in post_links[:5]:
        print(f"https://www.instagram.com{link}")
else:
    print("❌ No post links found. Instagram may have blocked scraping or changed its structure.")

#🔹 Step 4: Visit Each Post & Extract Image URL, Date, Location, and Text
data_list = []  # Store post details for Excel

# Test post ids
# post_links_test = ["/p/CtkZcXfN762/", "/p/CC85HWKs9HN/", "/p/B4Hg8y3lUNi/", "/p/BoxIOXUA-C_/", "/p/B3PEbB3hbkt/"]
for i, post_link in enumerate(post_links[:50]):  # Limit to 5 posts for now: post_links[:5]
    post_url = f"https://www.instagram.com{post_link}"
    print(f"\n🔗 Visiting Post: {post_url}")
    
    driver.get(post_url)
    time.sleep(5)

    post_soup = BeautifulSoup(driver.page_source, "html.parser")

    # 🔹 Skip video posts — video posts contain a <video> element, photo posts do not
    if post_soup.find("video"):
        print(f"⏭️ Skipping video post: {post_link}")
        continue

    # 🔹 Extract Image URL (note: does not work correctly for video posts)
    # img_tag = post_soup.find("meta", property="og:image")
    img_tags = post_soup.find_all("img", class_="x5yr21d")
    # First x5yr21d is the profile pic (when logged in), second is the post image
    img_tag = img_tags[1] if len(img_tags) > 1 else (img_tags[0] if img_tags else None)
    # img_url = img_tag["content"] if img_tag else "❌ No Image Found"
    img_url = img_tag["src"] if img_tag and "src" in img_tag.attrs else "❌ No Image Found"

    # 🔹 Extract Post Info
    description_tag = post_soup.find("meta", property="og:description")
    post_description = description_tag["content"] if description_tag else "❌ No Date Found"
    username, date, content = parse_description(post_description)

    # Get Location
    location = get_location(post_soup)

    # 🔹 Step 5: Download Image
    if img_url != "❌ No Image Found":
        img_response = requests.get(img_url, stream=True)
        if img_response.status_code == 200:
            img_filename = os.path.join(HASHTAG, f"image_{i+1}.jpg")  # Save inside "gibraltarmacaques"

            # 🔹 Save Data to List
            data_list.append([date, username, location, content, img_filename, img_url, post_url])
            
            with open(img_filename, "wb") as file:
                for chunk in img_response.iter_content(1024):
                    file.write(chunk)
            print(f"✅ Image saved: {img_filename}")
        else:
            print("❌ Failed to download image.")

# Close the browser
driver.quit()

# 🔹 Step 6: Save Data to Excel
df = pd.DataFrame(data_list, columns=["Date", "Username", "Location", "Content", "Image Name", "Image URL", "Post URL"])
excel_filename = "instagram_posts.xlsx"
df.to_excel(excel_filename, index=False)

print("🎉 Scraping complete!")



#!/usr/bin/env python3
"""
Austria News Agent

An AI-powered news scanning agent that collects and filters news stories relevant to
a 62-year-old American male living in Western Austria, interested in local news,
healthcare, retirement, American expat services, outdoor activities, and cultural events.

The agent fetches news from various sources, analyzes them for relevance, and generates
a daily summary of interesting stories tailored to the user's specific situation.
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import schedule
import nltk
from nltk.tokenize import sent_tokenize
from newspaper import Article
from dotenv import load_dotenv

# Load environment variables from .env file
# Try to load from the current directory first
if not load_dotenv():
    # If that fails, try the absolute path
    load_dotenv(dotenv_path="/home/frank/IP_Agent-1/.env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AustriaNewsAgent")

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class NewsSource:
    """Base class for news sources"""
    def __init__(self, name: str, language: str = "en"):
        self.name = name
        self.language = language
    
    def fetch_articles(self) -> List[Dict[str, Any]]:
        """Fetch articles from the source"""
        raise NotImplementedError("Subclasses must implement fetch_articles")

class NewsAPISource(NewsSource):
    """News source using NewsAPI"""
    def __init__(self, name: str, api_key: str = "d21636b5fd9941baaf25380d6e42bc65", country: str = "at", language: str = "en", endpoint: str = "top-headlines"):
        super().__init__(name, language)
        self.api_key = api_key
        self.country = country
        self.endpoint = endpoint
        self.base_url = f"https://newsapi.org/v2/{endpoint}"
    
    def fetch_articles(self, categories: List[str] = None) -> List[Dict[str, Any]]:
        """Fetch articles from NewsAPI"""
        all_articles = []
        
        # Default categories if none provided
        if not categories:
            categories = ["business", "entertainment", "general", "health", "science", "sports", "technology"]
        
        for category in categories:
            # Different parameters based on endpoint
            if self.endpoint == "top-headlines":
                params = {
                    "country": self.country,
                    "category": category,
                    "apiKey": self.api_key,
                    "language": self.language,
                    "pageSize": 100
                }
            else:  # everything endpoint
                params = {
                    "q": f"Austria OR {category}",  # Search for Austria or the category
                    "apiKey": self.api_key,
                    "language": self.language,
                    "pageSize": 100,
                    "sortBy": "publishedAt"
                }
            
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") == "ok":
                    articles = data.get("articles", [])
                    for article in articles:
                        article["source_name"] = self.name
                        article["category"] = category
                    all_articles.extend(articles)
                    logger.info(f"Fetched {len(articles)} articles from {self.name} in category {category}")
                else:
                    logger.error(f"Error fetching from {self.name}: {data.get('message', 'Unknown error')}")
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error fetching from {self.name}: {str(e)}")
            
            # Respect API rate limits
            time.sleep(1)
        
        return all_articles

class GuardianSource(NewsSource):
    """News source using The Guardian API"""
    def __init__(self, name: str, api_key: str = None, language: str = "en"):
        super().__init__(name, language)
        self.api_key = api_key
        self.base_url = "https://content.guardianapis.com/search"
    
    def fetch_articles(self, topics: List[str] = None) -> List[Dict[str, Any]]:
        """Fetch articles from The Guardian API"""
        all_articles = []
        
        # Default topics if none provided
        if not topics:
            topics = [
                "Austria healthcare", "Austria retirement", "American expat Austria", 
                "Vorarlberg", "Bregenz", "Dornbirn", "Feldkirch", "Bludenz",
                "Lake Constance", "Bodensee", "Bregenzerwald", "Montafon", "Arlberg",
                "Austria hiking", "Austria skiing", "Austria outdoor", "Austria festival",
                "US citizen abroad", "Medicare abroad", "Social security abroad"
            ]
        
        for topic in topics:
            params = {
                "q": topic,
                "lang": self.language,
                "page-size": 10,  # Number of articles to fetch
                "order-by": "newest",
                "show-fields": "headline,trailText,body,byline,publication",
                "api-key": self.api_key if self.api_key else ""
            }
            
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "response" in data and "results" in data["response"]:
                    articles = data["response"]["results"]
                    processed_articles = []
                    
                    for article in articles:
                        # Convert Guardian format to match our standard format
                        processed = {
                            "title": article.get("webTitle", ""),
                            "url": article.get("webUrl", ""),
                            "publishedAt": article.get("webPublicationDate", ""),
                            "source": {"name": "The Guardian"},
                            "source_name": self.name,
                            "category": topic,
                        }
                        
                        # Add fields if available
                        fields = article.get("fields", {})
                        if fields:
                            processed["description"] = fields.get("trailText", "")
                            processed["content"] = fields.get("body", "")
                            processed["author"] = fields.get("byline", "")
                        
                        processed_articles.append(processed)
                    
                    all_articles.extend(processed_articles)
                    logger.info(f"Fetched {len(processed_articles)} articles from {self.name} for topic {topic}")
                else:
                    logger.error(f"Error fetching from {self.name}: {data.get('message', 'Unknown error')}")
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error fetching from {self.name}: {str(e)}")
            
            # Respect API rate limits
            time.sleep(1)
        
        return all_articles

class GermanNewsSource(NewsSource):
    """News source for German language content from Austria"""
    def __init__(self, name: str, language: str = "de"):
        super().__init__(name, language)
        self.base_url = "https://www.derstandard.at/rss"
    
    def fetch_articles(self, categories: List[str] = None) -> List[Dict[str, Any]]:
        """Fetch articles from Austrian news RSS feeds"""
        import feedparser
        from bs4 import BeautifulSoup
        import html
        
        all_articles = []
        
        # Default RSS feeds if none provided - focusing on Vorarlberg
        feeds = [
            "https://www.vol.at/rss/feed/rss.xml",  # Vorarlberg Online (main Vorarlberg news)
            "https://www.vorarlberg.at/rss",        # Official Vorarlberg government news
            "https://www.vienna.at/rss/vorarlberg", # Vienna.at Vorarlberg section
            "https://www.orf.at/rss/vorarlberg",    # ORF Vorarlberg
            "https://www.vn.at/rss"                 # Vorarlberger Nachrichten
        ]
        
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                source_name = feed.feed.title if hasattr(feed, 'feed') and hasattr(feed.feed, 'title') else "Austrian News"
                
                for entry in feed.entries[:10]:  # Limit to 10 articles per feed
                    # Clean up the content
                    summary = ""
                    if hasattr(entry, 'summary'):
                        # Remove HTML tags
                        soup = BeautifulSoup(entry.summary, "html.parser")
                        summary = soup.get_text()
                    elif hasattr(entry, 'description'):
                        soup = BeautifulSoup(entry.description, "html.parser")
                        summary = soup.get_text()
                    
                    # Unescape HTML entities
                    summary = html.unescape(summary)
                    
                    article = {
                        "title": entry.title if hasattr(entry, 'title') else "",
                        "url": entry.link if hasattr(entry, 'link') else "",
                        "publishedAt": entry.published if hasattr(entry, 'published') else "",
                        "source": {"name": source_name},
                        "source_name": self.name,
                        "category": "Local News",
                        "description": summary,
                        "content": summary,
                        "author": entry.author if hasattr(entry, 'author') else ""
                    }
                    
                    all_articles.append(article)
                
                logger.info(f"Fetched {len(feed.entries[:10])} articles from {source_name}")
                
            except Exception as e:
                logger.error(f"Error fetching from {feed_url}: {str(e)}")
            
            # Respect rate limits
            time.sleep(1)
        
        return all_articles

class MediaStackSource(NewsSource):
    """News source using MediaStack API"""
    def __init__(self, name: str, api_key: str = "9e8f8b8e8f8b8e8f8b8e8f8b8e8f8b8e", language: str = "en"):
        super().__init__(name, language)
        self.api_key = api_key
        self.base_url = "http://api.mediastack.com/v1/news"
    
    def fetch_articles(self, topics: List[str] = None) -> List[Dict[str, Any]]:
        """Fetch articles from MediaStack API"""
        all_articles = []
        
        # Default topics if none provided
        if not topics:
            topics = [
                "Austria healthcare", "Austria retirement", "American expat Austria", 
                "Vorarlberg", "Bregenz", "Dornbirn", "Feldkirch", "Bludenz",
                "Lake Constance", "Bodensee", "Bregenzerwald", "Montafon", "Arlberg",
                "Austria hiking", "Austria skiing", "Austria outdoor", "Austria festival",
                "US citizen abroad", "Medicare abroad", "Social security abroad"
            ]
        
        for topic in topics:
            params = {
                "access_key": self.api_key,
                "keywords": topic,
                "languages": self.language,
                "limit": 10,
                "sort": "published_desc"
            }
            
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "data" in data:
                    articles = data["data"]
                    processed_articles = []
                    
                    for article in articles:
                        # Convert MediaStack format to match our standard format
                        processed = {
                            "title": article.get("title", ""),
                            "url": article.get("url", ""),
                            "publishedAt": article.get("published_at", ""),
                            "source": {"name": article.get("source", "MediaStack")},
                            "source_name": self.name,
                            "category": topic,
                            "description": article.get("description", ""),
                            "content": article.get("description", ""),  # MediaStack doesn't provide full content
                            "author": article.get("author", "")
                        }
                        
                        processed_articles.append(processed)
                    
                    all_articles.extend(processed_articles)
                    logger.info(f"Fetched {len(processed_articles)} articles from {self.name} for topic {topic}")
                else:
                    logger.error(f"Error fetching from {self.name}: {data.get('error', {}).get('message', 'Unknown error')}")
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error fetching from {self.name}: {str(e)}")
            
            # Respect API rate limits
            time.sleep(1)
        
        return all_articles

class AustrianNewsFilter:
    """Filter for news relevant to an American expat in Western Austria"""
    
    def __init__(self):
        # Keywords for different interest categories
        self.healthcare_keywords = [
            "healthcare", "health insurance", "krankenversicherung", "doctor", "arzt", "hospital", 
            "krankenhaus", "medical", "medizinisch", "prescription", "rezept", "pharmacy", "apotheke",
            "senior health", "elderly care", "health services", "health benefits", "medicare abroad",
            "health coverage", "specialist", "facharzt", "emergency care", "preventive care"
        ]
        
        self.retirement_keywords = [
            "retirement", "pension", "rente", "social security", "retirement visa", "retirement benefits",
            "retirement planning", "retirement community", "senior living", "retirement income",
            "401k abroad", "IRA international", "expat retirement", "retirement tax", "pension tax",
            "retirement healthcare", "senior activities", "senior services", "aging abroad"
        ]
        
        self.expat_keywords = [
            "expat", "american abroad", "american expat", "us citizen abroad", "expatriate", 
            "foreign resident", "residency permit", "aufenthaltstitel", "visa", "embassy", "consulate",
            "american community", "english speaking", "international community", "expat services",
            "expat tax", "FATCA", "FBAR", "dual citizenship", "passport renewal", "consular services"
        ]
        
        self.outdoor_keywords = [
            "hiking", "wandern", "skiing", "skifahren", "mountain biking", "mountainbike", "cycling",
            "radfahren", "fishing", "angeln", "golf", "swimming", "schwimmen", "outdoor activities",
            "nature", "natur", "trail", "wanderweg", "lake", "see", "mountain", "berg", "alps", "alpen",
            "national park", "nationalpark", "forest", "wald", "river", "fluss"
        ]
        
        self.cultural_keywords = [
            "festival", "concert", "konzert", "exhibition", "ausstellung", "museum", "gallery", "galerie",
            "theater", "theatre", "opera", "oper", "cinema", "kino", "music", "musik", "art", "kunst",
            "performance", "cultural event", "kulturveranstaltung", "traditional", "folk", "volksfest",
            "local customs", "local traditions", "heritage", "history", "geschichte"
        ]
        
        self.vorarlberg_keywords = [
            "vorarlberg", "bregenz", "dornbirn", "feldkirch", "bludenz", "hohenems", "lustenau", 
            "rankweil", "götzis", "hard", "lauterach", "wolfurt", "höchst", "altach", "lochau",
            "bregenzerwald", "montafon", "bodensee", "lake constance", "rheintal", "rhine valley",
            "arlberg", "silvretta", "brandnertal", "klostertal", "großes walsertal", "kleinwalsertal",
            "walgau", "leiblachtal", "laternsertal", "bodensee-vorarlberg", "alpenrhein", "ill river",
            "bregenzer festspiele", "bregenz festival", "schubertiade", "poolbar festival", "bezau",
            "mellau", "damüls", "schröcken", "warth", "lech", "zürs", "stuben", "schruns", "tschagguns",
            "gaschurn", "st. gallenkirch", "bartholomäberg", "vandans", "nenzing", "frastanz", "satteins"
        ]
        
        # Combined keywords for initial filtering
        self.all_keywords = (
            self.healthcare_keywords + 
            self.retirement_keywords + 
            self.expat_keywords + 
            self.outdoor_keywords + 
            self.cultural_keywords + 
            self.vorarlberg_keywords
        )
    
    def is_relevant(self, article: Dict[str, Any]) -> bool:
        """Check if an article is relevant based on title, description, and recency using a scoring system"""
        # Extract text from article
        title = article.get("title", "").lower()
        description = article.get("description", "").lower()
        content = article.get("content", "").lower()
        
        # Combine all text for searching
        all_text = f"{title} {description} {content}"
        
        # Filter out "as it happened" articles and other live/outdated content
        live_content_patterns = [
            "as it happened", "live updates", "live blog", "live coverage", 
            "minute by minute", "breaking news", "latest updates",
            "live: ", "live – ", "live - ", "happening now", "developing story",
            "updates: ", "follow live", "live report", "live reaction",
            "live news", "live stream", "live feed"
        ]
        
        # Check title for live content patterns
        if any(pattern in title.lower() for pattern in live_content_patterns):
            logger.info(f"Filtering out live/outdated article: {title}")
            return False
            
        # Check URL for live blog indicators
        url = article.get("url", "").lower()
        if url and ("/live/" in url or "-live-" in url or "/liveblog/" in url or "-liveblog-" in url):
            logger.info(f"Filtering out live blog article based on URL: {title}")
            return False
            
        # Filter out articles that are clearly about other countries and not Austria
        non_austria_patterns = [
            "in the uk", "uk this", "uk's", "britain", "british", 
            "england", "scotland", "wales", "northern ireland", 
            "london", "manchester", "edinburgh", "glasgow", "cardiff",
            "in france", "in italy", "in spain", "in germany", 
            "in the us", "in america", "american cities", "us cities"
        ]
        
        # Check if the article is specifically about another country and doesn't mention Austria
        if any(pattern in title.lower() for pattern in non_austria_patterns):
            # Only filter if Austria or Vorarlberg is not also mentioned
            if not any(keyword in title.lower() for keyword in ["austria", "austrian", "vorarlberg"]):
                logger.info(f"Filtering out non-Austria article: {title}")
                return False
        
        # Check if the article is recent (within the last 3 days - stricter time filter)
        published_at = article.get("publishedAt", "")
        if published_at:
            try:
                # Parse the date from the article using various formats
                pub_date = None
                
                # Try ISO format first
                if 'T' in published_at:
                    try:
                        pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    except ValueError:
                        pass
                
                # Try RSS/RFC 2822 format (e.g., 'Thu, 25 Sep 2025 03:00:00 +0000')
                if not pub_date and ',' in published_at:
                    try:
                        from email.utils import parsedate_to_datetime
                        pub_date = parsedate_to_datetime(published_at)
                    except Exception:
                        pass
                
                # Try other common formats
                if not pub_date:
                    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%a %b %d %H:%M:%S %z %Y"]:
                        try:
                            pub_date = datetime.strptime(published_at, fmt)
                            break
                        except ValueError:
                            continue
                
                # If we successfully parsed the date, check if it's older than 3 days (stricter)
                if pub_date:
                    days_old = (datetime.now() - pub_date).days
                    if days_old > 3:
                        logger.info(f"Filtering out old article from {pub_date.strftime('%Y-%m-%d')} ({days_old} days old): {title}")
                        return False
                else:
                    logger.warning(f"Could not parse date '{published_at}' for article '{title}' with any known format")
            except Exception as e:
                # If we can't parse the date, just continue with other checks
                logger.warning(f"Error processing date '{published_at}' for article '{title}': {str(e)}")
        
        # Implement a scoring system for relevance
        relevance_score = 0
        
        # Check for Vorarlberg-specific keywords in title (highest priority)
        if any(keyword.lower() in title.lower() for keyword in self.vorarlberg_keywords):
            relevance_score += 5
            
        # Check for other category keywords in title
        if any(keyword.lower() in title.lower() for keyword in self.healthcare_keywords):
            relevance_score += 3
        if any(keyword.lower() in title.lower() for keyword in self.retirement_keywords):
            relevance_score += 3
        if any(keyword.lower() in title.lower() for keyword in self.expat_keywords):
            relevance_score += 3
            
        # For outdoor activities and cultural events, require stronger Austria connection
        # to avoid articles about hiking trails in the UK, etc.
        if any(keyword.lower() in title.lower() for keyword in self.outdoor_keywords):
            # Check if there's an Austria connection
            if any(keyword in all_text.lower() for keyword in ["austria", "austrian", "vorarlberg", "tyrol", "tirol", "alps", "alpen"]):
                relevance_score += 3  # Higher score if Austria-related
            else:
                relevance_score += 1  # Lower score if not clearly Austria-related
                
        if any(keyword.lower() in title.lower() for keyword in self.cultural_keywords):
            # Check if there's an Austria connection
            if any(keyword in all_text.lower() for keyword in ["austria", "austrian", "vorarlberg", "vienna", "wien", "salzburg"]):
                relevance_score += 3  # Higher score if Austria-related
            else:
                relevance_score += 1  # Lower score if not clearly Austria-related
            
        # Check for Vorarlberg-specific keywords in description/content
        if any(keyword.lower() in description.lower() or keyword.lower() in content.lower() 
               for keyword in self.vorarlberg_keywords):
            relevance_score += 3
            
        # Check for other category keywords in description/content
        if any(keyword.lower() in description.lower() or keyword.lower() in content.lower() 
               for keyword in self.healthcare_keywords):
            relevance_score += 2
        if any(keyword.lower() in description.lower() or keyword.lower() in content.lower() 
               for keyword in self.retirement_keywords):
            relevance_score += 2
        if any(keyword.lower() in description.lower() or keyword.lower() in content.lower() 
               for keyword in self.expat_keywords):
            relevance_score += 2
        if any(keyword.lower() in description.lower() or keyword.lower() in content.lower() 
               for keyword in self.outdoor_keywords):
            relevance_score += 1
        if any(keyword.lower() in description.lower() or keyword.lower() in content.lower() 
               for keyword in self.cultural_keywords):
            relevance_score += 1
            
        # Count the number of unique keywords that match
        matched_keywords = set()
        for keyword in self.all_keywords:
            if keyword.lower() in all_text:
                matched_keywords.add(keyword.lower())
        
        # Add points based on the number of unique keywords matched
        if len(matched_keywords) >= 3:
            relevance_score += 3
        elif len(matched_keywords) == 2:
            relevance_score += 1
            
        # Require a minimum score to consider the article relevant
        # Higher threshold means fewer articles will pass the filter
        min_score = 6  # Increased threshold to be more selective
        
        is_relevant = relevance_score >= min_score
        
        if not is_relevant:
            logger.info(f"Filtering out low-relevance article (score {relevance_score}): {title}")
            
        return is_relevant
    
    def categorize_article(self, article: Dict[str, Any]) -> Dict[str, bool]:
        """Categorize article by interest areas"""
        # Extract text from article
        title = article.get("title", "").lower()
        description = article.get("description", "").lower()
        content = article.get("content", "").lower()
        
        # Combine all text for searching
        all_text = f"{title} {description} {content}"
        
        # Check each category
        categories = {
            "Healthcare": any(keyword.lower() in all_text for keyword in self.healthcare_keywords),
            "Retirement": any(keyword.lower() in all_text for keyword in self.retirement_keywords),
            "American Expat": any(keyword.lower() in all_text for keyword in self.expat_keywords),
            "Outdoor Activities": any(keyword.lower() in all_text for keyword in self.outdoor_keywords),
            "Cultural Events": any(keyword.lower() in all_text for keyword in self.cultural_keywords),
            "Vorarlberg": any(keyword.lower() in all_text for keyword in self.vorarlberg_keywords)
        }
        
        return categories

class ArticleProcessor:
    """Process and enrich article data"""
    
    def __init__(self):
        self.translator = None
        try:
            from googletrans import Translator
            self.translator = Translator()
        except ImportError:
            logger.warning("Googletrans not available. German articles will not be translated.")
        except Exception as e:
            logger.warning(f"Error initializing translator: {str(e)}")
    
    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enrich an article with full content and summary"""
        processed = article.copy()
        
        # Get the article URL
        url = article.get("url")
        if not url:
            return processed
        
        try:
            # Use newspaper3k to extract full article content
            news_article = Article(url)
            news_article.download()
            news_article.parse()
            news_article.nlp()  # This generates summary, keywords, etc.
            
            # Add extracted data to the processed article
            processed["full_text"] = news_article.text
            processed["summary"] = news_article.summary
            processed["keywords"] = news_article.keywords
            processed["processed_date"] = datetime.now().isoformat()
            
            # Translate German content if needed
            if self.translator and article.get("language", "") == "de":
                try:
                    # Translate title
                    if processed.get("title"):
                        translated_title = self.translator.translate(processed["title"], src="de", dest="en")
                        processed["original_title"] = processed["title"]
                        processed["title"] = f"{translated_title.text} [Translated from German]"
                    
                    # Translate summary
                    if processed.get("summary"):
                        translated_summary = self.translator.translate(processed["summary"], src="de", dest="en")
                        processed["original_summary"] = processed["summary"]
                        processed["summary"] = translated_summary.text
                    
                    logger.info(f"Translated German article: {processed.get('title', 'Unknown title')}")
                except Exception as e:
                    logger.error(f"Error translating article: {str(e)}")
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing article {url}: {str(e)}")
            return processed

class NewsDigestGenerator:
    """Generate a formatted digest of news articles"""
    
    def __init__(self):
        pass
    
    def generate_digest(self, articles: List[Dict[str, Any]], categories: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate a formatted digest of news articles"""
        today = datetime.now().strftime("%A, %B %d, %Y")
        
        digest = f"# Daily News Digest for American Expat in Vorarlberg\n\n"
        digest += f"## {today}\n\n"
        
        # Add summary statistics
        total_articles = sum(len(cat) for cat in categories.values())
        digest += f"Found {total_articles} relevant articles today.\n\n"
        
        # Add articles by category, prioritizing Vorarlberg news
        # Show Vorarlberg news first
        priority_categories = ["Vorarlberg", "American Expat", "Healthcare", "Retirement", 
                              "Outdoor Activities", "Cultural Events"]
        
        # Track articles we've already included to avoid duplicates
        included_urls = set()
        included_titles = set()
        
        for category in priority_categories:
            cat_articles = categories.get(category, [])
            if cat_articles:
                # Filter out articles that have already been included in other categories
                # Check both URL and title to avoid duplicates
                unique_articles = [a for a in cat_articles 
                                  if a.get("url", "") not in included_urls 
                                  and a.get("title", "").strip() not in included_titles]
                
                if unique_articles:
                    digest += f"## {category.title()} News ({len(unique_articles)} articles)\n\n"
                    
                    # Limit to top 3 per category for a more concise digest
                    articles_to_show = min(3, len(unique_articles))
                    for article in unique_articles[:articles_to_show]:
                        title = article.get("title", "No title")
                        source = article.get("source", {}).get("name", article.get("source_name", "Unknown source"))
                        url = article.get("url", "#")
                        summary = article.get("summary", article.get("description", "No summary available"))
                        
                        # Truncate summary if it's too long
                        if summary and len(summary) > 300:
                            summary = summary[:297] + "..."
                        
                        digest += f"### [{title}]({url})\n"
                        digest += f"**Source:** {source}\n\n"
                        digest += f"{summary}\n\n"
                        digest += "---\n\n"
                        
                        # Add this URL and title to our tracking sets
                        included_urls.add(article.get("url", ""))
                        included_titles.add(article.get("title", "").strip())
        
        return digest

class AustriaNewsAgent:
    """Main agent class that orchestrates the news scanning process"""
    
    def __init__(self, news_api_key: str):
        # Get Guardian API key from environment (optional)
        guardian_api_key = os.getenv("GUARDIAN_API_KEY", "")
        
        # Get MediaStack API key from environment (optional)
        mediastack_api_key = os.getenv("MEDIASTACK_API_KEY", "9e8f8b8e8f8b8e8f8b8e8f8b8e8f8b8e")
        
        self.news_sources = [
            # Use The Guardian API (works without API key but with limitations)
            GuardianSource("Guardian", api_key=guardian_api_key, language="en"),
            
            # Use MediaStack API (requires API key but we provide a default one)
            MediaStackSource("MediaStack", api_key=mediastack_api_key, language="en"),
            
            # Add German language sources for local news
            GermanNewsSource("Austrian Local News", language="de"),
            
            # Add more news sources as needed
        ]
        
        self.filter = AustrianNewsFilter()
        self.processor = ArticleProcessor()
        self.digest_generator = NewsDigestGenerator()
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
    
    def run_daily_scan(self):
        """Run a complete daily news scan"""
        logger.info("Starting daily news scan")
        
        # 1. Fetch articles from all sources
        all_articles = []
        for source in self.news_sources:
            articles = source.fetch_articles()
            all_articles.extend(articles)
        
        logger.info(f"Fetched {len(all_articles)} articles in total")
        
        # 2. Filter for relevant articles
        relevant_articles = [article for article in all_articles if self.filter.is_relevant(article)]
        logger.info(f"Found {len(relevant_articles)} relevant articles")
        
        # Sort articles by date (newest first) if available
        def get_article_date(article):
            published_at = article.get("publishedAt", "")
            if not published_at:
                return datetime.min.replace(tzinfo=None)  # Default to oldest date if no date available
                
            try:
                # Try different date formats
                if 'T' in published_at:
                    dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    # Convert to naive datetime for consistent comparison
                    return dt.replace(tzinfo=None)
                elif ',' in published_at:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(published_at)
                    # Convert to naive datetime for consistent comparison
                    return dt.replace(tzinfo=None)
                else:
                    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%a %b %d %H:%M:%S %z %Y"]:
                        try:
                            dt = datetime.strptime(published_at, fmt)
                            # Convert to naive datetime if it has timezone info
                            if dt.tzinfo is not None:
                                return dt.replace(tzinfo=None)
                            return dt
                        except ValueError:
                            continue
            except Exception as e:
                logger.debug(f"Date parsing error for '{published_at}': {str(e)}")
                
            return datetime.min.replace(tzinfo=None)  # Default to oldest date if parsing fails
        
        # Sort by date, newest first
        relevant_articles.sort(key=get_article_date, reverse=True)
        
        # Limit to a maximum of 30 articles total
        max_articles = 30
        if len(relevant_articles) > max_articles:
            logger.info(f"Limiting from {len(relevant_articles)} to {max_articles} articles")
            relevant_articles = relevant_articles[:max_articles]
        
        # 3. Process and enrich articles
        processed_articles = []
        for article in relevant_articles:
            processed = self.processor.process_article(article)
            processed_articles.append(processed)
        
        # 4. Categorize articles
        categorized = {
            "Healthcare": [],
            "Retirement": [],
            "American Expat": [],
            "Outdoor Activities": [],
            "Cultural Events": [],
            "Vorarlberg": []
        }
        
        for article in processed_articles:
            categories = self.filter.categorize_article(article)
            for category, relevant in categories.items():
                if relevant:
                    categorized[category].append(article)
        
        # Limit the number of articles per category to ensure balance
        max_per_category = 5
        for category in categorized:
            if len(categorized[category]) > max_per_category:
                categorized[category] = categorized[category][:max_per_category]
                logger.info(f"Limited {category} category to {max_per_category} articles")
        
        # 5. Generate digest
        digest = self.digest_generator.generate_digest(processed_articles, categorized)
        
        # 6. Save results
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Save raw articles
        with open(f"data/articles_{today}.json", "w") as f:
            json.dump(processed_articles, f, indent=2)
        
        # Save digest
        with open(f"data/digest_{today}.md", "w") as f:
            f.write(digest)
        
        logger.info(f"Saved digest to data/digest_{today}.md")
        
        return digest

def main():
    """Main function to run the agent"""
    # Get API key from environment variable or use the hardcoded one
    default_api_key = "d21636b5fd9941baaf25380d6e42bc65"
    news_api_key = os.getenv("NEWS_API_KEY", default_api_key)
    
    # If the API key is still the placeholder, use the hardcoded one
    if news_api_key == "your_api_key_here":
        news_api_key = default_api_key
        
    print(f"Using API key: {news_api_key}")
    
    if not news_api_key:
        logger.error("NEWS_API_KEY environment variable not set")
        print("Please set the NEWS_API_KEY environment variable")
        print("You can get a free API key from https://newsapi.org/")
        return
    
    # Create agent
    agent = AustriaNewsAgent(news_api_key)
    
    # Run once immediately
    agent.run_daily_scan()
    
    # Schedule daily runs
    schedule.every().day.at("07:00").do(agent.run_daily_scan)
    
    # Keep running
    print("Vorarlberg News Agent for American Expats is running.")
    print("Fetching news relevant to healthcare, retirement, American expat services,")
    print("outdoor activities, cultural events, and Vorarlberg local news.")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("News agent stopped.")

if __name__ == "__main__":
    main()
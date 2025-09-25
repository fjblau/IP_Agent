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
        # Prioritize vol.at as requested by user
        feeds = [
            "https://www.vol.at/rss/feed/rss.xml",  # Vorarlberg Online (main Vorarlberg news) - PRIMARY SOURCE
            "https://www.vol.at/rss/feed/vorarlberg.xml",  # VOL.at Vorarlberg specific feed
            "https://www.vol.at/rss/feed/oesterreich.xml", # VOL.at Austria news
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
            
        # Filter out articles that are clearly about non-EU/US countries
        # Focus on EU and US as per user preference
        non_relevant_patterns = [
            # Non-EU/US regions
            "africa", "african", "malawi", "philippines", "duterte", "hong kong",
            "china", "chinese", "india", "indian", "pakistan", "middle east",
            "brazil", "brazilian", "mexico", "mexican", "australia", "australian", 
            "new zealand", "japan", "japanese", "russia", "russian", 
            "indonesia", "malaysia", "thailand", "vietnam", "korea",
            "bangladesh", "nigeria", "egypt", "kenya",
            "argentina", "colombia", "peru", "chile", "venezuela"
        ]
        
        # Countries that should be excluded unless explicitly about Austria
        # Keep EU countries and US in the list but with lower priority
        excluded_countries = [
            # Non-EU/US countries (high priority exclusion)
            "china", "india", "japan", "russia", "brazil", "mexico", 
            "australia", "new zealand", "africa", "philippines", 
            "indonesia", "malaysia", "thailand", "vietnam", "korea",
            "pakistan", "bangladesh", "nigeria", "egypt", "south africa", "kenya",
            "argentina", "colombia", "peru", "chile", "venezuela",
            
            # EU/US countries (lower priority exclusion - only exclude if no Austria connection)
            "uk", "britain", "england", "scotland", "wales", "france", "italy", "spain",
            "germany", "us", "usa", "america", "canada"
        ]
        
        # Check if the article is specifically about a non-relevant country
        if any(pattern in title.lower() for pattern in non_relevant_patterns):
            # Only filter if Austria or Vorarlberg is not also mentioned
            if not any(keyword in title.lower() for keyword in ["austria", "austrian", "vorarlberg", "eu", "europe", "european union"]):
                logger.info(f"Filtering out non-relevant article: {title}")
                return False
                
        # Check if the article mentions a country that's not Austria
        # and doesn't explicitly mention Austria in the title or first part of content
        for country in excluded_countries:
            if country in title.lower() or country in description.lower()[:100]:
                # Only filter if Austria or Vorarlberg is not also prominently mentioned
                if not any(keyword in title.lower() or keyword in description.lower()[:100] 
                          for keyword in ["austria", "austrian", "vorarlberg", "vienna", "wien", "salzburg", "tyrol", "tirol"]):
                    logger.info(f"Filtering out article about {country}: {title}")
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
                    # Ensure both datetimes are timezone-naive for comparison
                    if pub_date.tzinfo is not None:
                        pub_date = pub_date.replace(tzinfo=None)
                    
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
            relevance_score += 7  # Increased from 5 to prioritize Vorarlberg content
            
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
                relevance_score += 4  # Higher score if Austria-related (increased from 3)
            else:
                relevance_score += 0  # No points if not clearly Austria-related (decreased from 1)
                
        if any(keyword.lower() in title.lower() for keyword in self.cultural_keywords):
            # Check if there's an Austria connection
            if any(keyword in all_text.lower() for keyword in ["austria", "austrian", "vorarlberg", "vienna", "wien", "salzburg"]):
                relevance_score += 4  # Higher score if Austria-related (increased from 3)
            else:
                relevance_score += 0  # No points if not clearly Austria-related (decreased from 1)
        
        # Bonus points for Austrian sources
        url = article.get("url", "").lower()
        if any(domain in url for domain in ["vol.at", "vienna.at", "vorarlberg.at", "orf.at", "vn.at"]):
            relevance_score += 3  # Bonus for Austrian news sources
            
        # Check for Vorarlberg-specific keywords in description/content
        if any(keyword.lower() in description.lower() or keyword.lower() in content.lower() 
               for keyword in self.vorarlberg_keywords):
            relevance_score += 4  # Increased from 3 to prioritize Vorarlberg content
            
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
            # Only add points if there's an Austria connection
            if any(keyword in all_text.lower() for keyword in ["austria", "austrian", "vorarlberg", "tyrol", "tirol", "alps", "alpen"]):
                relevance_score += 2  # Increased from 1 for Austria-related outdoor content
        if any(keyword.lower() in description.lower() or keyword.lower() in content.lower() 
               for keyword in self.cultural_keywords):
            # Only add points if there's an Austria connection
            if any(keyword in all_text.lower() for keyword in ["austria", "austrian", "vorarlberg", "vienna", "wien", "salzburg"]):
                relevance_score += 2  # Increased from 1 for Austria-related cultural content
            
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
        """Categorize article by interest areas with improved accuracy"""
        # Extract text from article
        title = article.get("title", "").lower()
        description = article.get("description", "").lower()
        content = article.get("content", "").lower()
        
        # Combine all text for searching
        all_text = f"{title} {description} {content}"
        
        # Prioritize title matches (3x weight)
        title_weight = 3
        
        # Calculate scores for each category
        vorarlberg_score = sum(1 * title_weight if keyword.lower() in title else 1 
                              for keyword in self.vorarlberg_keywords 
                              if keyword.lower() in all_text)
        
        healthcare_score = sum(1 * title_weight if keyword.lower() in title else 1 
                              for keyword in self.healthcare_keywords 
                              if keyword.lower() in all_text)
        
        retirement_score = sum(1 * title_weight if keyword.lower() in title else 1 
                              for keyword in self.retirement_keywords 
                              if keyword.lower() in all_text)
        
        expat_score = sum(1 * title_weight if keyword.lower() in title else 1 
                         for keyword in self.expat_keywords 
                         if keyword.lower() in all_text)
        
        outdoor_score = sum(1 * title_weight if keyword.lower() in title else 1 
                           for keyword in self.outdoor_keywords 
                           if keyword.lower() in all_text)
        
        cultural_score = sum(1 * title_weight if keyword.lower() in title else 1 
                            for keyword in self.cultural_keywords 
                            if keyword.lower() in all_text)
        
        # Require a minimum score to consider an article part of a category
        # This prevents articles with just one minor keyword mention from being categorized
        min_category_score = 2
        
        # Special case: If article is specifically about Austria/Vorarlberg, lower the threshold
        is_austria_specific = any(keyword in title.lower() for keyword in ["austria", "austrian", "vorarlberg", "vienna", "wien"])
        if is_austria_specific:
            min_category_score = 1
        
        # Check if article is from an Austrian source
        url = article.get("url", "").lower()
        is_austrian_source = any(domain in url for domain in ["vol.at", "vienna.at", "vorarlberg.at", "orf.at", "vn.at"])
        
        # Determine if article has Austria/Vorarlberg connection
        has_austria_connection = any(keyword in all_text.lower() for keyword in 
                                    ["austria", "austrian", "vorarlberg", "tyrol", "tirol", 
                                     "vienna", "wien", "salzburg", "innsbruck", "graz", "linz"])
        
        # Special case for Vorarlberg - prioritize this category
        is_vorarlberg_specific = vorarlberg_score >= 2 or "vorarlberg" in title.lower() or is_austrian_source
        
        # Check each category with minimum score requirement and appropriate context
        categories = {
            "Healthcare": healthcare_score >= min_category_score and (has_austria_connection or is_austrian_source),
            "Retirement": retirement_score >= min_category_score and (has_austria_connection or is_austrian_source),
            "American Expat": expat_score >= min_category_score,
            "Outdoor Activities": outdoor_score >= min_category_score and (has_austria_connection or is_austrian_source),
            "Cultural Events": cultural_score >= min_category_score and (has_austria_connection or is_austrian_source),
            "Vorarlberg": is_vorarlberg_specific  # Special handling for Vorarlberg as primary focus
        }
        
        # If article is from an Austrian source but doesn't fit any category, put it in Vorarlberg
        if is_austrian_source and not any(categories.values()):
            categories["Vorarlberg"] = True
        
        return categories

class ArticleProcessor:
    """Process and enrich article data"""
    
    def __init__(self):
        self.translator = None
        try:
            from googletrans import Translator
            # Initialize with specific service URLs to improve reliability
            self.translator = Translator(service_urls=['translate.google.com', 'translate.google.co.kr'])
            # Test the translator with a simple phrase
            test_translation = self.translator.translate('Hallo', src='de', dest='en')
            logger.info(f"Translator initialized successfully. Test: 'Hallo' -> '{test_translation.text}'")
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
            
            # Detect if content is in German (either from source language or content analysis)
            is_german = article.get("language", "") == "de" or "vol.at" in url.lower()
            
            # Also check the domain for Austrian sources
            austrian_domains = ["vol.at", "vienna.at", "vorarlberg.at", "orf.at", "vn.at", "derstandard.at"]
            if any(domain in url.lower() for domain in austrian_domains):
                processed["is_austrian_source"] = True
                # If not explicitly marked as German but from Austrian source, check content
                if not is_german and processed.get("full_text"):
                    # Simple heuristic: check for common German words
                    german_indicators = ["und", "der", "die", "das", "für", "mit", "von", "bei", "ist"]
                    text_sample = processed["full_text"][:500].lower()
                    if sum(1 for word in german_indicators if f" {word} " in text_sample) >= 3:
                        is_german = True
                        processed["language"] = "de"
            
            # Translate German content if needed
            if self.translator and is_german:
                try:
                    # Translate title
                    if processed.get("title"):
                        try:
                            translated_title = self.translator.translate(processed["title"], src="de", dest="en")
                            processed["original_title"] = processed["title"]
                            processed["title"] = f"{translated_title.text} [Translated from German]"
                            logger.info(f"Translated title: '{processed['original_title']}' -> '{translated_title.text}'")
                        except Exception as e:
                            logger.warning(f"Error translating title: {str(e)}")
                    
                    # Translate summary
                    if processed.get("summary"):
                        try:
                            translated_summary = self.translator.translate(processed["summary"], src="de", dest="en")
                            processed["original_summary"] = processed["summary"]
                            processed["summary"] = translated_summary.text
                        except Exception as e:
                            logger.warning(f"Error translating summary: {str(e)}")
                    
                    # Translate description if available
                    if processed.get("description"):
                        try:
                            translated_desc = self.translator.translate(processed["description"], src="de", dest="en")
                            processed["original_description"] = processed["description"]
                            processed["description"] = translated_desc.text
                        except Exception as e:
                            logger.warning(f"Error translating description: {str(e)}")
                    
                    # Translate a portion of the full text if available
                    if processed.get("full_text") and len(processed["full_text"]) > 0:
                        try:
                            # Only translate the first 1000 characters to avoid API limits
                            text_to_translate = processed["full_text"][:1000]
                            translated_text = self.translator.translate(text_to_translate, src="de", dest="en")
                            processed["translated_text_sample"] = translated_text.text
                            if not processed.get("summary") or len(processed["summary"]) < 50:
                                # Use translated text as summary if no good summary exists
                                processed["summary"] = translated_text.text[:300] + "..."
                        except Exception as e:
                            logger.warning(f"Error translating text sample: {str(e)}")
                    
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
                unique_articles = []
                for a in cat_articles:
                    url = a.get("url", "").strip()
                    title = a.get("title", "").strip()
                    # Remove the "[Translated from German]" suffix for comparison
                    title_for_comparison = title.replace(" [Translated from German]", "").strip()
                    
                    if url and url not in included_urls and title_for_comparison not in included_titles:
                        unique_articles.append(a)
                        included_urls.add(url)
                        included_titles.add(title_for_comparison)
                
                if unique_articles:
                    digest += f"## {category.title()} News ({len(unique_articles)} articles)\n\n"
                    
                    # Limit to top 5 per category as requested by user
                    articles_to_show = min(5, len(unique_articles))
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
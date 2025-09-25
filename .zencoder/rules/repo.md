---
description: Repository Information Overview
alwaysApply: true
---

# Austria News Agent Information

## Summary
The Austria News Agent is a specialized news aggregation tool designed for an American expat living in Western Austria (Vorarlberg). It collects, filters, and summarizes news stories relevant to the user's specific situation, focusing on local news, healthcare, retirement, American expat services, outdoor activities, and cultural events in Austria.

## Structure
- `austria_news_agent.py`: Main application file containing all functionality
- `data/`: Directory storing fetched articles and generated digests
- `requirements.txt`: Python dependencies
- `.env`: Environment variables configuration (API keys)

## Language & Runtime
**Language**: Python
**Version**: Python 3.12
**Build System**: None (single script application)
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- requests==2.31.0: HTTP requests library
- nltk==3.8.1: Natural language processing toolkit
- newspaper3k==0.2.8: News article extraction
- schedule==1.2.0: Job scheduling
- feedparser==6.0.10: RSS feed parsing
- beautifulsoup4==4.12.2: HTML parsing
- googletrans==4.0.0-rc1: Translation services

## Build & Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run the agent
python austria_news_agent.py
```

## Key Components

### News Sources
- **NewsAPISource**: Fetches articles from NewsAPI
- **GuardianSource**: Fetches articles from The Guardian API
- **GermanNewsSource**: Fetches German-language content from Austrian RSS feeds (VOL.at, ORF, etc.)
- **MediaStackSource**: Fetches articles from MediaStack API

### Content Processing
- **ArticleProcessor**: Extracts full content, generates summaries, and translates German content
- **RelevanceFilter**: Scores and filters articles based on relevance to an American expat in Vorarlberg
- **ArticleCategorizer**: Categorizes articles into topics like Local News, Healthcare, Outdoor Activities

### Output Generation
- **DigestGenerator**: Creates a daily digest of relevant news in markdown format
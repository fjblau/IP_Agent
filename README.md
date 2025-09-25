# Austria News Agent for Expats

An AI-powered news scanning agent that collects and filters news stories relevant to an Austrian expat interested in tax issues, employment, music, and literature.

## Features

- Daily scanning of news sources for relevant articles
- Filtering based on personalized interest areas:
  - Tax issues relevant to expats
  - Employment and retirement information
  - Music events and news
  - Literature and book-related content
- Automatic categorization of articles
- Daily digest generation in markdown format
- Scheduled runs with customizable timing

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Get a free API key from [NewsAPI](https://newsapi.org/)
4. Create a `.env` file based on `.env.example` and add your API key:
   ```
   NEWS_API_KEY=your_api_key_here
   ```

## Usage

Run the agent manually:
```
python austria_news_agent.py
```

The agent will:
1. Fetch news articles from various sources
2. Filter for relevant content based on your interests
3. Process and categorize the articles
4. Generate a daily digest in markdown format
5. Save both raw article data and the formatted digest in the `data` directory

By default, the agent will run once immediately and then schedule daily runs at 7:00 AM.

## Customization

You can customize the agent by modifying:
- Keywords for each interest category in `AustrianNewsFilter` class
- News sources in the `AustriaNewsAgent` initialization
- Scheduling time in the `main()` function
- Digest format in the `NewsDigestGenerator` class
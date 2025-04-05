# function_app.py
import azure.functions as func
import logging
import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import AzureOpenAI
import tweepy
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
import pytz
from postgrest.exceptions import APIError

# Load environment variables
load_dotenv()

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Configuration
SUPABASE_TABLE_NAME = os.getenv("SUPABASE_TABLE_NAME", "articles")  # Configurable table name
DEFAULT_WINDOW_MINUTES = 60  # Fallback window for first run

# Global variable to track last trigger time
LAST_TRIGGER_TIME = None

class APIClientFactory:
    """Factory class to initialize API clients."""
    
    @staticmethod
    def create_supabase_client() -> Client:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("Supabase URL and KEY must be set in environment variables")
        return create_client(url, key)

    @staticmethod
    def create_openai_client() -> AzureOpenAI:
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    @staticmethod
    def create_twitter_client() -> tweepy.Client:
        return tweepy.Client(
            consumer_key=os.getenv("CONSUMER_KEY", ""),
            consumer_secret=os.getenv("CONSUMER_SECRET", ""),
            access_token=os.getenv("ACCESS_TOKEN", ""),
            access_token_secret=os.getenv("ACCESS_TOKEN_SECRET", ""),
            bearer_token=os.getenv("BEARER_TOKEN", "")
        )

class DataProcessor:
    """Handles data fetching, preprocessing, and filtering."""

    def __init__(self, supabase_client: Client):
        self.supabase_client = supabase_client

    def fetch_new_articles(self, last_trigger_time: Optional[datetime]) -> pd.DataFrame:
        """Fetches new articles since last trigger time in GMT."""
        gmt = pytz.timezone('GMT')
        
        if last_trigger_time is None:
            # First run - get articles from default window
            last_trigger_time = datetime.now(gmt) - timedelta(minutes=DEFAULT_WINDOW_MINUTES)
        
        logging.info(f"Fetching articles from table '{SUPABASE_TABLE_NAME}' created after {last_trigger_time} GMT")
        
        try:
            query = self.supabase_client.table(SUPABASE_TABLE_NAME)\
                      .select("*")\
                      .gt("created_at", last_trigger_time.isoformat())
            
            data = query.execute()
            df = pd.DataFrame(data.data) if data.data else pd.DataFrame()
            logging.info(f"Retrieved {len(df)} new records")
            return df
            
        except APIError as e:
            logging.error(f"Supabase API Error: {str(e)}")
            raise ValueError(f"Could not access table '{SUPABASE_TABLE_NAME}'. Please verify the table exists.") from e
        except Exception as e:
            logging.error(f"Error fetching articles: {str(e)}")
            raise

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and standardizes data with GMT timezone."""
        if df.empty:
            logging.warning("Empty DataFrame received for preprocessing")
            return df
            
        original_count = len(df)
        df.replace("", np.nan, inplace=True)
        
        # Convert to GMT timezone
        gmt = pytz.timezone('GMT')
        if 'created_at' in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"]).dt.tz_convert(gmt)
        if 'date' in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(gmt)
        
        logging.info(f"Preprocessed {original_count} records")
        return df

    @staticmethod
    def remove_flagged_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicate entries based on 'company' flag."""
        if df.empty:
            return df
            
        original_count = len(df)
        if "company" in df.columns:
            df = df.loc[~df["company"].astype(str).str.startswith("Duplicate of")].copy()
        logging.info(f"Deduplication: {len(df)}/{original_count} records remain")
        return df

    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Handles missing values in the dataset."""
        if df.empty:
            return df
            
        original_count = len(df)
        df = df.copy()
        
        if 'article_type' in df.columns:
            df = df[~df["article_type"].isin(["None of the Above", "None of the Above from Link"])]
        if 'company' in df.columns:
            df = df[~df["company"].isin(["None of the Above"])]
            
        df.fillna({
            "summary": "Summary not provided",
            "company": "Unknown Company",
            "article_type": "Uncategorized",
            "sentiment": "Neutral",
        }, inplace=True)
        
        logging.info(f"Missing value handling: {len(df)}/{original_count} records remain")
        return df

class LLMAnalyzer:
    """Handles interaction with OpenAI LLM to analyze news impact."""

    def __init__(self, openai_client: AzureOpenAI):
        self.openai_client = openai_client

    def analyze_batch(self, batch_df: pd.DataFrame) -> Dict[str, int]:
        """Sends a batch of articles to LLM and gets impact scores."""
        batch_json = batch_df.to_json(orient="records", force_ascii=False)
        batch_json = json.loads(batch_json)

        prompt = f"""
        You are an AI stock news analyst. Evaluate the impact of the following stock news articles.
        Articles: {json.dumps(batch_json, indent=2)}

        Respond with a JSON object:
        {{
            "id": impact_score (1-10),
            "id": impact_score,
            ...
        }}
        Impact scores should consider:
        - Company significance
        - News urgency
        - Potential market impact
        - Novelty of information
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Error in LLM call: {e}")
            return {str(article_id): 1 for article_id in batch_df["id"]}

    def analyze_articles(self, df: pd.DataFrame, batch_size: int = 5) -> pd.DataFrame:
        """Processes articles in batches using the LLM model."""
        if df.empty:
            logging.warning("No articles to analyze")
            return df
            
        impact_scores = {}
        total_batches = (len(df) // batch_size) + 1
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            columns_to_drop = [col for col in ["created_at", "date", "link"] if col in batch_df.columns]
            batch_df = batch_df.drop(columns=columns_to_drop, errors="ignore")
            
            logging.info(f"Processing batch {i // batch_size + 1}/{total_batches}")
            batch_results = self.analyze_batch(batch_df)
            impact_scores.update(batch_results)
            time.sleep(1)  # Rate limiting

        df["impact_score"] = df["id"].astype(str).map(impact_scores)
        logging.info(f"Impact scores:\n{df['impact_score'].value_counts().sort_index()}")
        return df

class TwitterBot:
    """Handles article selection and posting to Twitter."""

    @staticmethod
    def recommend_articles(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
        """Filters articles based on impact_score threshold."""
        if df.empty:
            return df
            
        recommended = df[df["impact_score"] >= threshold]
        logging.info(f"Recommended {len(recommended)}/{len(df)} articles")
        if not recommended.empty:
            logging.info(f"Top recommendations:\n{recommended[['company', 'impact_score']].to_string()}")
        return recommended

    @staticmethod
    def convert_to_json_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Converts DataFrame to a list of JSON objects."""
        columns = [col for col in ['company', 'sentiment', 'summary', 'link', 'impact_score'] if col in df.columns]
        return df[columns].to_dict(orient="records")

    def post_articles_to_twitter(self, twitter_client: tweepy.Client, articles: List[Dict[str, str]]) -> None:
        """Posts articles to Twitter."""
        if not articles:
            logging.warning("No articles to post")
            return
            
        logging.info(f"Posting {len(articles)} articles")
        
        for idx, article in enumerate(articles, 1):
            try:
                required_fields = ['company', 'sentiment', 'summary', 'link']
                if not all(article.get(field) for field in required_fields):
                    logging.error(f"Skipping article {idx}: Missing fields")
                    continue

                base_text = f"""📢 {article['company']}
                Sentiment: {article['sentiment'].capitalize()}
                Summary: {article['summary']}
                Read more: {article['link']}"""
                
                tweet_text = ' '.join(base_text.split()).strip()
                if len(tweet_text) > 280:
                    tweet_text = tweet_text[:274] + "..."

                logging.info(f"Posting ({idx}/{len(articles)}): {tweet_text[:50]}...")
                response = twitter_client.create_tweet(text=tweet_text)
                logging.info(f"Posted tweet ID: {response.data['id']}")
                time.sleep(10)
                
            except tweepy.TweepyException as e:
                logging.error(f"Twitter error: {str(e)}")
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")

def run_pipeline(last_trigger_time: Optional[datetime]) -> datetime:
    """Main execution pipeline with dynamic window."""
    try:
        gmt = pytz.timezone('GMT')
        current_time = datetime.now(gmt)
        logging.info(f"\n{'='*40}\nStarting pipeline at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n{'='*40}")
        
        # Initialize clients
        supabase_client = APIClientFactory.create_supabase_client()
        openai_client = APIClientFactory.create_openai_client()
        twitter_client = APIClientFactory.create_twitter_client()

        # Initialize processors
        processor = DataProcessor(supabase_client)
        llm_analyzer = LLMAnalyzer(openai_client)
        twitter_bot = TwitterBot()

        # Fetch and process articles
        df = processor.fetch_new_articles(last_trigger_time)
        df = processor.preprocess_data(df)
        df = processor.remove_flagged_duplicates(df)
        df = processor.handle_missing_values(df)

        if df.empty:
            logging.warning("No new articles to process")
            return current_time  # Return current time as last processed

        # Analyze and post
        df = llm_analyzer.analyze_articles(df)
        recommended = twitter_bot.recommend_articles(df)
        
        if not recommended.empty:
            articles = twitter_bot.convert_to_json_list(recommended)
            twitter_bot.post_articles_to_twitter(twitter_client, articles)

        logging.info(f"\n{'='*40}\nPipeline completed at {datetime.now(gmt).strftime('%Y-%m-%d %H:%M:%S %Z')}\n{'='*40}")
        return current_time  # Return the time this run started
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return last_trigger_time  # On failure, don't update last trigger time

# HTTP Trigger
@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Starting pipeline via HTTP trigger')
    try:
        new_trigger_time = run_pipeline(None)  # None forces default window
        return func.HttpResponse(
            f"Pipeline executed at {new_trigger_time}\n"
            f"Table used: {SUPABASE_TABLE_NAME}",
            status_code=200
        )
    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        return func.HttpResponse(
            f"Error: {str(e)}\n"
            f"Table attempted: {SUPABASE_TABLE_NAME}",
            status_code=500
        )

# Timer Trigger
@app.schedule(schedule="0 */10 * * * *", arg_name="mytimer", run_on_startup=False)
def timer_trigger(mytimer: func.TimerRequest) -> None:
    global LAST_TRIGGER_TIME
    
    gmt = pytz.timezone('GMT')
    current_time = datetime.now(gmt)
    
    # Active hours: 8:00 AM to 3:40 PM IST (2:30-10:10 GMT)
    ist = pytz.timezone('Asia/Kolkata')
    current_ist = current_time.astimezone(ist)
    
    active_start = current_ist.replace(hour=8, minute=0, second=0, microsecond=0)
    active_end = current_ist.replace(hour=15, minute=40, second=0, microsecond=0)
    
    # Determine if we should run
    should_run = False
    if active_start <= current_ist <= active_end:
        logging.info(f'Active hours execution at {current_time.strftime("%Y-%m-%d %H:%M:%S %Z")}')
        should_run = True
    elif current_time.minute % 30 == 0:
        logging.info(f'30-min interval execution at {current_time.strftime("%Y-%m-%d %H:%M:%S %Z")}')
        should_run = True
    
    if should_run:
        LAST_TRIGGER_TIME = run_pipeline(LAST_TRIGGER_TIME)
    else:
        logging.info(f'Skipping execution at {current_time.strftime("%Y-%m-%d %H:%M:%S %Z")}')
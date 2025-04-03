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
from supabase import create_client
from typing import List, Dict, Any
import pytz

# Load environment variables
load_dotenv()

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

class APIClientFactory:
    """Factory class to initialize API clients."""
    
    @staticmethod
    def create_supabase_client() -> Any:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        return create_client(url, key)

    @staticmethod
    def create_openai_client() -> AzureOpenAI:
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
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

    def __init__(self, supabase_client: Any):
        self.supabase_client = supabase_client

    def fetch_supabase_data(self, table_name: str) -> pd.DataFrame:
        """Fetches data from Supabase."""
        data = self.supabase_client.table(table_name).select("*").execute()
        return pd.DataFrame(data.data) if data.data else pd.DataFrame()

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and standardizes data."""
        df.replace("", np.nan, inplace=True)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def filter_by_time(df: pd.DataFrame, start_time: datetime, interval_minutes: int = 10) -> pd.DataFrame:
        """Filters data based on time window."""
        start_window = start_time - timedelta(minutes=interval_minutes)
        return df[(df["created_at"] >= start_window) & (df["created_at"] < start_time)]

    @staticmethod
    def remove_flagged_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicate entries based on 'company' flag."""
        if "company" in df.columns:
            return df.loc[~df["company"].astype(str).str.startswith("Duplicate of")].copy()
        return df

    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Handles missing values in the dataset."""
        df = df.copy()
        df = df[~df["article_type"].isin(["None of the Above", "None of the Above from Link"])]
        df = df[~df["company"].isin(["None of the Above"])]
        df.fillna({
            "summary": "Summary not provided",
            "company": "Unknown Company",
            "article_type": "Uncategorized",
            "sentiment": "Neutral",
        }, inplace=True)
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
            "id": impact_score,
            "id": impact_score,
            ...
        }}
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Error in LLM call: {e}")
            return {str(article_id): 1 for article_id in batch_df["id"]}

    def analyze_articles(self, df: pd.DataFrame, batch_size: int = 10) -> pd.DataFrame:
        """Processes articles in batches using the LLM model."""
        impact_scores = {}
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size].drop(columns=["created_at", "date", "link"], errors="ignore")
            logging.info(f"Processing batch {i // batch_size + 1}...")
            batch_results = self.analyze_batch(batch_df)
            impact_scores.update(batch_results)
            time.sleep(1)

        df["impact_score"] = df["id"].astype(str).map(impact_scores)
        return df

class TwitterBot:
    """Handles article selection and posting to Twitter."""

    @staticmethod
    def recommend_articles(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
        """Filters articles based on impact_score threshold."""
        return df[df["impact_score"] >= threshold]

    @staticmethod
    def convert_to_json_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Converts DataFrame to a list of JSON objects."""
        return df[['company', 'sentiment', 'summary', 'link']].to_dict(orient="records")

    def post_articles_to_twitter(self, twitter_client: tweepy.Client, articles: List[Dict[str, str]]) -> None:
        """Posts articles to Twitter."""
        for article in articles:
            base_text = f"""📢 {article['company']}
            Sentiment: {article['sentiment'].capitalize()}
            Summary: {article['summary']}
            Read more: {article['link']}"""
            
            tweet_text = ' '.join(base_text.split()).strip()
            if len(tweet_text) > 280:
                tweet_text = tweet_text[:274] + "..."

            try:
                response = twitter_client.create_tweet(text=tweet_text)
                logging.info("-"*20)
                logging.info(f"Tweet posted successfully: {response.data}")
            except tweepy.TweepyException as e:
                logging.error(f"Failed to post tweet for {article['company']}: {e}")

def run_pipeline() -> None:
    """Main execution pipeline that integrates all steps."""
    try:
        # Capture start_time FIRST in IST
        ist = pytz.timezone('Asia/Kolkata')
        start_time = datetime.now(ist)
        
        # Initialize API clients
        supabase_client = APIClientFactory.create_supabase_client()
        openai_client = APIClientFactory.create_openai_client()
        twitter_client = APIClientFactory.create_twitter_client()

        # Initialize processing modules
        data_processor = DataProcessor(supabase_client)
        llm_analyzer = LLMAnalyzer(openai_client)
        twitter_bot = TwitterBot()

        # Step 1: Fetch and preprocess data
        df = data_processor.fetch_supabase_data("articles_rows_calquity")
        df = data_processor.preprocess_data(df)

        # Step 2: Filter and clean articles
        filtered_articles = data_processor.filter_by_time(df, start_time)
        cleaned_articles = data_processor.remove_flagged_duplicates(filtered_articles)
        processed_articles = data_processor.handle_missing_values(cleaned_articles)

        # Step 3: Analyze articles using LLM
        processed_articles = llm_analyzer.analyze_articles(processed_articles)

        # Step 4: Select and post top articles
        recommended_articles = twitter_bot.recommend_articles(processed_articles)
        article_list = twitter_bot.convert_to_json_list(recommended_articles)
        twitter_bot.post_articles_to_twitter(twitter_client, article_list)

        logging.info(f"Pipeline execution completed at {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

# HTTP Trigger
@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Starting pipeline via HTTP trigger')
    try:
        run_pipeline()
        return func.HttpResponse("Pipeline executed successfully", status_code=200)
    except Exception as e:
        logging.error(f"Error in pipeline execution: {str(e)}")
        return func.HttpResponse(f"Error occurred: {str(e)}", status_code=500)

# Timer Trigger with conditional execution
@app.schedule(schedule="0 */10 * * * *", arg_name="mytimer", run_on_startup=False)
def timer_trigger(mytimer: func.TimerRequest) -> None:
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    # Active hours: 8:00 AM to 3:40 PM IST
    active_start = current_time.replace(hour=8, minute=0, second=0, microsecond=0)
    active_end = current_time.replace(hour=15, minute=40, second=0, microsecond=0)
    
    # Determine execution frequency
    if active_start <= current_time <= active_end:
        # Every 10 minutes during active hours
        logging.info(f'Executing 10-min interval pipeline at {current_time.strftime("%Y-%m-%d %H:%M:%S %Z")}')
        run_pipeline()
    elif current_time.minute % 30 == 0:  # :00 and :30 minutes
        # Every 30 minutes outside active hours
        logging.info(f'Executing 30-min interval pipeline at {current_time.strftime("%Y-%m-%d %H:%M:%S %Z")}')
        run_pipeline()
    else:
        logging.info(f'Skipping execution at {current_time.strftime("%Y-%m-%d %H:%M:%S %Z")}')
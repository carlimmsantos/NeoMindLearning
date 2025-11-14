import logging
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import re

logger = logging.getLogger(__name__)


class DataProcessor: 
    _data: pd.DataFrame

    def __init__(self, data_path: str = "data"):
        """
        Initialize with data directory path.

        Args:
            data_path (str): Path to the directory where data files are stored.
        """
        self._data_path = Path(data_path)
        self._data_path.mkdir(exist_ok=True)
        self._data = None
    
    def _create_sample_comment_data(self) -> pd.DataFrame:
        """Create sample customer comment data for the assignment."""
        np.random.seed(42)  # For reproducible results
        
        categories = ["category A", "category B", "category C", "category D"]
        sentiments = ["positive", "negative", "neutral"]
        
        # Sample comment texts with varying sentiments
        comment_samples = {
            "positive": [
                "I absolutely love this category! It exceeded my expectations.",
                "Great quality and fast shipping. Highly recommend!",
                "Excellent customer service and fantastic category quality.",
                "This category changed my life. Worth every penny!",
                "Amazing features and user-friendly interface. Five stars!",
                "Best purchase I've made this year. Outstanding quality!",
                "Impressed with the build quality and performance.",
                "Great value for money. Will definitely buy again.",
            ],
            "negative": [
                "Terrible category. Waste of money. Would not recommend.",
                "Poor quality materials. Broke after one week of use.",
                "Worst customer service ever. Still waiting for my refund.",
                "category doesn't match the description. Very disappointed.",
                "Overpriced for what you get. There are better alternatives.",
                "Shipping took forever and category arrived damaged.",
                "Difficult to use and confusing instructions.",
                "Not worth the money. Cheaply made and unreliable.",
            ],
            "neutral": [
                "It's an okay category. Does what it's supposed to do.",
                "Average quality. Nothing special but gets the job done.",
                "The category works fine. No major complaints.",
                "Decent category for the price. Could be better.",
                "It's alright. Met my basic expectations.",
                "Good enough for what I needed. Standard quality.",
                "The category is functional but not impressive.",
                "Fair price for an average category. Nothing outstanding.",
            ]
        }
        
        data = []
        for i in range(500):  # Generate 500 comment entries
            sentiment = np.random.choice(sentiments, p=[0.4, 0.3, 0.3])  # Slightly more positive
            category = np.random.choice(categories)
            comment = np.random.choice(comment_samples[sentiment])
            rating = {
                "positive": np.random.randint(4, 5),
                "negative": np.random.randint(1, 2),
                "neutral": 3
            }[sentiment]
            
            data.append({
                "title": f"Sample Comment {i + 1}",
                "category": category,
                "comment": comment,
                "rating": rating,
                "date": pd.date_range(start="2023-01-01", end="2024-01-01", periods=500)[i]
            })
        
        return pd.DataFrame(data)

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the dataset.
        """
        summary = {
            "total_records": len(df),
            "categories": df['category'].value_counts().to_dict() if 'category' in df.columns else {},
            "rating_distribution": df['rating'].value_counts().to_dict() if 'rating' in df.columns else {},
            "average_rating": df['rating'].mean() if 'rating' in df.columns else None,
            "date_range": {
                "start": df['date'].min().isoformat() if 'date' in df.columns else None,
                "end": df['date'].max().isoformat() if 'date' in df.columns else None
            } if 'date' in df.columns else None
        }
        
        return summary

    def load_customer_comments(self, filename: str = "customer_comment.csv") -> pd.DataFrame:
        """
        Load customer comment dataset.
        """
        df = pd.read_csv(filename, encoding="latin-1")
        df = self._clean_comments_data(df)
        return df
    
    def _clean_comments_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the comments data.

        This includes:
        - Removing duplicates
        - Handling missing values
        - Cleaning text data
        - Ensuring numeric columns are properly formatted
        - Converting date columns to datetime format if present

        Args:
            df (pd.DataFrame): The DataFrame containing comments data.

        Returns:
            pd.DataFrame: Cleaned DataFrame ready for analysis.
        """

        # Remove duplicates
        df = df.drop_duplicates().reset_index(drop=True)

        # Drop unnecessary columns
        if "Customer name" in df.columns:
            df = df.drop(columns=["Customer name"], axis=1)
        
        # Rename columns for consistency
        df = df.rename(columns={
            "Review Title": "title",
            "Rating": "rating",
            "Date": "date",
            "Category": "category",
            "Comments": "comment", 
        })


        # Clean text data
        df['comment'] = df['comment'].str.strip().str.lower()

        # Ensure numeric columns are properly formatted
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.dropna(subset=['rating'])

        # Convert date columns to datetime format if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])

        return df
    
    def _clean_useful_column(self, df: pd.DataFrame, column_name: str = "useful") -> pd.DataFrame:
        """
        Clean the 'useful' column in the dataset.

        Examples:
            - "7 people found this helpful" -> 7
            - "One person found this helpful" -> 1
            - "" or Nan -> 0 
     
        """

        if "useful" not in df.columns:
            return df
        
        def extract_useful_count(value: str) -> int:
            '''
            Extract the number of people who found the comment useful.
            '''

            if pd.isna(value):
                return 0
            
            value_str = str(value).lower().strip()

            if not value_str:
                return 0

            value_str = re.sub(r'[Oo]ne\s+person.*', '1', value_str)

            match = re.search(r'(\d+)', value_str)

            return int(match.group(1)) if match else 0

        df[column_name] = df[column_name].apply(extract_useful_count).astype(int)

        return df
    
    def _clean_rating_column(self, df: pd.DataFrame, column_name: str = "rating" ) -> pd.DataFrame:
        """
        Clean the 'rating' column to ensure it contains numeric values only.

        Examples:
            - "4.0 out of 5 stars" -> 4.0
            - "5.0 out of 5 stars" -> 5.0
        """

        if column_name not in df.columns:
            return df
        
        def extract_rating(value: str) -> float:
            '''
            Extract numeric rating from string.
            '''
            if pd.isna(value):
                return np.nan
            
            value_str = str(value).lower().strip()

            match = re.search(r'(\d+(\.\d+)?)', value_str)
            return float(match.group(1)) if match else np.nan
        
        df[column_name] = df[column_name].apply(extract_rating)

        df[column_name] = df[column_name].fillna(0).astype(float)

        return df

    def _clean_date_column(self, df: pd.DataFrame, column_name: str = "date") -> pd.DataFrame:
        """
        Clean the 'date' column to ensure it is in datetime format.
        """

        if column_name not in df.columns:
            return df
        
        def clean_date(value):
            '''Convert date string to datetime or return NaT'''

            if pd.isna(value):
                return pd.NaT
            
            date_str = str(value).strip().lower()

            if not date_str or date_str == 'nan':
                return pd.NaT
            
            try:
                pattern = r'on\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})'
                match = re.search(pattern, date_str)

                if match:
                    day = int(match.group(1))
                    month_str = match.group(2)
                    year = int(match.group(3))
                    

                    date_obj = pd.to_datetime(
                        f"{day} {month_str} {year}", 
                        format="%d %B %Y"
                    )
                    return date_obj
                else:
                    return pd.NaT
            
            except Exception as e:
                logger.warning(f"Error processing date '{date_str}': {e}")
                return pd.NaT
        
        df[column_name] = df[column_name].apply(clean_date)

        valid_dates = df[column_name].dropna()
        if len(valid_dates) > 0:
            logger.debug(f"Date range after cleaning: {valid_dates.min().date()} to {valid_dates.max().date()}")
        
        return df
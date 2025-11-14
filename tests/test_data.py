import pandas as pd
from src.data.processor import DataProcessor


class TestDataProcessor:
    """Test data processing functionality."""
    
    def test_create_sample_comment_data(self, temp_data_dir: str):
        """Test creating sample comment data."""
        processor = DataProcessor(temp_data_dir)
        sample_data = processor._create_sample_comment_data()
        
        assert isinstance(sample_data, pd.DataFrame)
        assert len(sample_data) == 500
        assert "title" in sample_data.columns
        assert "comment" in sample_data.columns
        assert "rating" in sample_data.columns
        assert "category" in sample_data.columns
        assert "date" in sample_data.columns
    
    def test_clean_comment_data(self, sample_comment_data: pd.DataFrame):
        """Test data cleaning functionality."""
        processor = DataProcessor()
        
        # Add some dirty data
        dirty_data = sample_comment_data.copy()
        cleaned_data = processor._clean_comments_data(dirty_data)
        
        assert {"title", "category", "comment", "rating", "date"}.issubset(set(cleaned_data.columns))
        assert cleaned_data["date"].dtype == "datetime64[ns]"
        assert cleaned_data["rating"].dtype == "int64"
    
    def test_get_data_summary(self, sample_comment_data: pd.DataFrame):
        """Test data summary generation."""
        processor = DataProcessor()
        summary = processor.get_data_summary(sample_comment_data)

        print(list(summary.keys()))  # Debugging line to check keys in summary
        
        assert "total_records" in summary
        assert "categories" in summary
        assert "rating_distribution" in summary
        assert "average_rating" in summary
        assert "date_range" in summary
        assert summary["total_records"] == 5
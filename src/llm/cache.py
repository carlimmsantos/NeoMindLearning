"""LLM Response Caching Module."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LLMCache:
    """Simple file-based cache for LLM responses to reduce API calls."""
    
    def __init__(self, cache_dir: str = ".cache/llm"):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        logger.info(f"LLMCache initialized at {self.cache_dir}")
    
    def _hash_prompt(self, prompt: str, model: str) -> str:
        """
        Create a hash key for a prompt.
        
        Args:
            prompt: The prompt text
            model: The model name
            
        Returns:
            MD5 hash of the combined prompt and model
        """
        key = f"{model}:{prompt}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, prompt: str, model: str) -> Optional[str]:
        """
        Retrieve a cached response.
        
        Args:
            prompt: The prompt text
            model: The model name
            
        Returns:
            Cached response content or None if not found
        """
        cache_file = self.cache_dir / f"{self._hash_prompt(prompt, model)}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.hits += 1
                    logger.debug(f"Cache HIT - Model: {model}")
                    return data.get("content")
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        self.misses += 1
        return None
    
    def set(self, prompt: str, model: str, response: str) -> None:
        """
        Cache a response.
        
        Args:
            prompt: The prompt text
            model: The model name
            response: The response content to cache
        """
        try:
            cache_file = self.cache_dir / f"{self._hash_prompt(prompt, model)}.json"
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "prompt_hash": self._hash_prompt(prompt, model),
                        "model": model,
                        "content": response
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            logger.debug(f"Response cached for model: {model}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hit/miss statistics
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        cache_size = len(list(self.cache_dir.glob("*.json")))
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": cache_size
        }
    
    def clear(self) -> None:
        """Clear all cached responses."""
        try:
            for file in self.cache_dir.glob("*.json"):
                file.unlink()
            self.hits = 0
            self.misses = 0
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def print_stats(self) -> None:
        """Print cache statistics to logger."""
        stats = self.stats()
        logger.info(f"Cache Stats - Hits: {stats['hits']}, Misses: {stats['misses']}, "
                   f"Hit Rate: {stats['hit_rate']}, Size: {stats['cache_size']} files")
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
        Initialize the LLM response cache with specified directory.
        
        Creates cache directory if it does not exist and initializes
        performance metrics counters.

        Args:
            cache_dir: Directory path to store cache files. Defaults to '.cache/llm'.
                      Created automatically if directory does not exist.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        logger.info(f"LLMCache initialized at {self.cache_dir}")
    
    def _hash_prompt(self, prompt: str, model: str) -> str:
        """
        Generate unique cache key from prompt and model identifier.
        
        Combines prompt text and model name into a single string,
        then hashes using MD5 algorithm for consistent key generation.
        This ensures deterministic cache lookup and avoids file system
        naming conflicts.

        Args:
            prompt: The prompt text submitted to the LLM
            model: The model identifier (e.g., 'gpt-4', 'claude-3')
            
        Returns:
            32-character MD5 hexadecimal hash string
        """
        key = f"{model}:{prompt}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, prompt: str, model: str) -> Optional[str]:
        """
        Retrieve cached LLM response if available.
        
        Attempts to load previously cached response from file system
        using the prompt-model hash as key. Tracks cache hit/miss
        statistics for performance monitoring.

        Args:
            prompt: The prompt text to lookup in cache
            model: The model identifier used to generate the response
            
        Returns:
            Cached response content as string, or None if not found
        """
        cache_file = self.cache_dir / f"{self._hash_prompt(prompt, model)}.json"
        
        if cache_file.exists():
            try:
                # Load cached JSON response from disk
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
        Store LLM response in cache for future retrieval.
        
        Persists response to JSON file using prompt-model hash as filename.
        Includes metadata (hash, model) alongside response content for
        debugging and audit purposes.

        Args:
            prompt: The original prompt text submitted to LLM
            model: The model identifier that generated the response
            response: The LLM response content to cache
        """
        try:
            cache_file = self.cache_dir / f"{self._hash_prompt(prompt, model)}.json"
            
            # Serialize cache entry with metadata
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
        Calculate and return cache performance statistics.
        
        Computes hit rate percentage, total operations count, and
        current cache size in files. Provides metrics for monitoring
        cache effectiveness and optimization opportunities.
        
        Returns:
            Dictionary with keys:
            - hits: Number of successful cache retrievals
            - misses: Number of cache misses
            - total: Total cache operations (hits + misses)
            - hit_rate: Percentage string of successful retrievals
            - cache_size: Number of cached files in directory
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
        """
        Purge all cached responses and reset performance counters.
        
        Deletes all JSON cache files from cache directory and resets
        hit/miss counters to zero. Useful for clearing stale cached
        responses or testing clean cache behavior.
        """
        try:
            # Delete all JSON files in cache directory
            for file in self.cache_dir.glob("*.json"):
                file.unlink()
            # Reset performance counters
            self.hits = 0
            self.misses = 0
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def print_stats(self) -> None:
        """
        Output cache statistics to logger for monitoring and debugging.
        
        Formats and logs cache performance metrics including hit count,
        miss count, hit rate percentage, and total cached file count.
        Useful for periodic performance reporting.
        """
        stats = self.stats()
        logger.info(f"Cache Stats - Hits: {stats['hits']}, Misses: {stats['misses']}, "
                   f"Hit Rate: {stats['hit_rate']}, Size: {stats['cache_size']} files")
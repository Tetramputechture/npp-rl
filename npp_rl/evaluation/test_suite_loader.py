"""
Test Suite Loader for NPP-RL Evaluation (Task 3.3).

This module provides utilities for loading and managing the comprehensive test suite
of 250 deterministic N++ levels across 5 complexity categories.

The test suite dataset is located in the nclone repository at:
    nclone/datasets/test_suite/

Usage:
    from npp_rl.evaluation.test_suite_loader import TestSuiteLoader
    
    # Load test suite (point to nclone dataset)
    loader = TestSuiteLoader('/path/to/nclone/datasets/test_suite')
    
    # Get all simple levels
    simple_levels = loader.get_category('simple')
    
    # Get a specific level
    level = loader.get_level('simple_000')
    
    # Load level into environment
    env.unwrapped.nplay_headless.load_map_from_map_data(level['map_data'])
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class TestSuiteLoader:
    """Loader for NPP-RL test suite maps."""
    
    CATEGORIES = ['simple', 'medium', 'complex', 'mine_heavy', 'exploration']
    
    def __init__(self, test_suite_dir: str):
        """Initialize the test suite loader.
        
        Args:
            test_suite_dir: Path to the test suite directory
        """
        self.test_suite_dir = Path(test_suite_dir)
        
        if not self.test_suite_dir.exists():
            raise ValueError(f"Test suite directory not found: {test_suite_dir}")
        
        # Load metadata
        metadata_file = self.test_suite_dir / 'test_suite_metadata.json'
        if not metadata_file.exists():
            raise ValueError(f"Test suite metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self._level_cache: Dict[str, Dict[str, Any]] = {}
    
    def get_category(self, category: str) -> List[Dict[str, Any]]:
        """Load all levels from a specific category.
        
        Args:
            category: Category name ('simple', 'medium', 'complex', 'mine_heavy', 'exploration')
        
        Returns:
            List of level dictionaries
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"Unknown category: {category}. Must be one of {self.CATEGORIES}")
        
        category_dir = self.test_suite_dir / category
        if not category_dir.exists():
            raise ValueError(f"Category directory not found: {category_dir}")
        
        levels = []
        level_ids = self.metadata['categories'][category]['level_ids']
        
        for level_id in level_ids:
            levels.append(self.get_level(level_id))
        
        return levels
    
    def get_level(self, level_id: str) -> Dict[str, Any]:
        """Load a specific level by ID.
        
        Args:
            level_id: Level ID (e.g., 'simple_000', 'mine_heavy_000')
        
        Returns:
            Level dictionary with map_data and metadata
        """
        # Check cache first
        if level_id in self._level_cache:
            return self._level_cache[level_id]
        
        # Determine category from level_id (handle multi-word categories like 'mine_heavy')
        category = None
        for cat in self.CATEGORIES:
            if level_id.startswith(cat + '_'):
                category = cat
                break
        
        if category is None:
            raise ValueError(f"Invalid level_id format: {level_id}")
        
        # Load level file
        level_file = self.test_suite_dir / category / f"{level_id}.pkl"
        if not level_file.exists():
            raise ValueError(f"Level file not found: {level_file}")
        
        with open(level_file, 'rb') as f:
            level = pickle.load(f)
        
        # Cache the level
        self._level_cache[level_id] = level
        
        return level
    
    def get_all_levels(self) -> List[Dict[str, Any]]:
        """Load all levels from all categories.
        
        Returns:
            List of all level dictionaries
        """
        all_levels = []
        for category in self.CATEGORIES:
            all_levels.extend(self.get_category(category))
        return all_levels
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get test suite metadata.
        
        Returns:
            Test suite metadata dictionary
        """
        return self.metadata
    
    def get_level_count(self, category: Optional[str] = None) -> int:
        """Get the number of levels in a category or total.
        
        Args:
            category: Optional category name. If None, returns total count.
        
        Returns:
            Number of levels
        """
        if category is None:
            return self.metadata['total_levels']
        
        if category not in self.CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        
        return self.metadata['categories'][category]['count']
    
    def get_level_ids(self, category: Optional[str] = None) -> List[str]:
        """Get list of level IDs.
        
        Args:
            category: Optional category name. If None, returns all level IDs.
        
        Returns:
            List of level IDs
        """
        if category is None:
            all_ids = []
            for cat in self.CATEGORIES:
                all_ids.extend(self.metadata['categories'][cat]['level_ids'])
            return all_ids
        
        if category not in self.CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        
        return self.metadata['categories'][category]['level_ids']


def load_test_suite_into_env(env, level_id: str, test_suite_dir: str) -> None:
    """Convenience function to load a test suite level into an environment.
    
    Args:
        env: NPP environment instance
        level_id: Level ID to load (e.g., 'simple_000')
        test_suite_dir: Path to test suite directory
    """
    loader = TestSuiteLoader(test_suite_dir)
    level = loader.get_level(level_id)
    env.unwrapped.nplay_headless.load_map_from_map_data(level['map_data'])


# Example usage
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m npp_rl.evaluation.test_suite_loader <test_suite_dir>")
        sys.exit(1)
    
    test_suite_dir = sys.argv[1]
    loader = TestSuiteLoader(test_suite_dir)
    
    print("=" * 70)
    print("NPP-RL Test Suite Information")
    print("=" * 70)
    print()
    print(f"Test suite directory: {test_suite_dir}")
    print(f"Total levels: {loader.get_level_count()}")
    print()
    print("Levels by category:")
    for category in loader.CATEGORIES:
        count = loader.get_level_count(category)
        print(f"  {category:15s}: {count:3d} levels")
    print()
    
    # Show a sample level
    print("Sample level (simple_000):")
    level = loader.get_level('simple_000')
    print(f"  Level ID: {level['level_id']}")
    print(f"  Category: {level['category']}")
    print(f"  Seed: {level['seed']}")
    print(f"  Description: {level['metadata']['description']}")
    print(f"  Difficulty tier: {level['metadata']['difficulty_tier']}")
    print(f"  Map data length: {len(level['map_data'])} bytes")
    print()
    print("=" * 70)

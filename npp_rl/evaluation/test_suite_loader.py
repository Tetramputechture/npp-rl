"""Test suite loader for standardized evaluation.

Loads and manages test levels from the nclone dataset structure.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class TestSuiteLoader:
    """Loads N++ test suite levels for evaluation."""

    CATEGORIES = [
        "simplest",
        "simplest_few_mines",
        "simplest_with_mines",
        "simpler",
        "simple",
        "medium",
        "complex",
        "exploration",
        "mine_heavy",
    ]

    def __init__(self, dataset_path: str):
        """Initialize test suite loader.

        Args:
            dataset_path: Path to dataset directory containing category subdirs
        """
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        self.metadata = self._load_metadata()
        logger.info(f"Initialized test suite loader: {self.dataset_path}")

    def _load_metadata(self) -> Optional[Dict]:
        """Load dataset metadata if available."""
        metadata_files = ["test_metadata.json", "train_metadata.json", "metadata.json"]

        for fname in metadata_files:
            metadata_path = self.dataset_path / fname
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from {fname}")
                return metadata

        logger.warning("No metadata file found")
        return None

    def load_category(self, category: str) -> List[Dict[str, Any]]:
        """Load all levels from a specific category.

        Args:
            category: Category name ('simple', 'medium', etc.)

        Returns:
            List of level data dictionaries
        """
        if category not in self.CATEGORIES:
            raise ValueError(
                f"Unknown category '{category}'. Valid categories: {self.CATEGORIES}"
            )

        category_dir = self.dataset_path / category

        if not category_dir.exists():
            logger.warning(f"Category directory not found: {category_dir}")
            return []

        levels = []

        for level_file in sorted(category_dir.glob("*.pkl")):
            try:
                with open(level_file, "rb") as f:
                    level_data = pickle.load(f)
                levels.append(level_data)
            except Exception as e:
                logger.error(f"Failed to load level {level_file}: {e}")

        logger.info(f"Loaded {len(levels)} levels from category '{category}'")
        return levels

    def load_all_levels(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all levels from all categories.

        Returns:
            Dictionary mapping category name to list of levels
        """
        all_levels = {}

        for category in self.CATEGORIES:
            all_levels[category] = self.load_category(category)

        total_count = sum(len(levels) for levels in all_levels.values())
        logger.info(
            f"Loaded {total_count} total levels across {len(all_levels)} categories"
        )

        return all_levels

    def get_level_count(self, category: Optional[str] = None) -> int:
        """Get count of levels in a category or total.

        Args:
            category: Category name, or None for total count

        Returns:
            Number of levels
        """
        if category is None:
            # Count all
            total = 0
            for cat in self.CATEGORIES:
                cat_dir = self.dataset_path / cat
                if cat_dir.exists():
                    total += len(list(cat_dir.glob("*.pkl")))
            return total
        else:
            cat_dir = self.dataset_path / category
            if cat_dir.exists():
                return len(list(cat_dir.glob("*.pkl")))
            return 0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of dataset.

        Returns:
            Summary dictionary with counts and metadata
        """
        summary = {
            "dataset_path": str(self.dataset_path),
            "categories": {},
            "total_levels": 0,
        }

        for category in self.CATEGORIES:
            count = self.get_level_count(category)
            summary["categories"][category] = count
            summary["total_levels"] += count

        if self.metadata:
            summary["metadata"] = self.metadata

        return summary

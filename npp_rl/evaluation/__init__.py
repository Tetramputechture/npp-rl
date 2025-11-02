"""Evaluation utilities for NPP-RL.

This package provides comprehensive evaluation tools including:
- Test suite loading and management
- Model evaluation on standardized test sets
- Performance metrics and analysis
"""

from npp_rl.evaluation.test_suite_loader import TestSuiteLoader
from npp_rl.evaluation.comprehensive_evaluator import ComprehensiveEvaluator

__all__ = [
    'TestSuiteLoader',
    'ComprehensiveEvaluator'
]

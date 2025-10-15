#!/usr/bin/env python3
"""
Data Quality and De-duplication Tools

This module provides tools for validating replay data quality and removing
duplicate trajectories from processed datasets.

Usage:
    python tools/data_quality.py --input datasets/processed --output quality_report.json
    python tools/data_quality.py --deduplicate datasets/processed --threshold 0.95
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for a dataset or trajectory."""
    total_samples: int
    valid_samples: int
    invalid_samples: int
    action_distribution: Dict[int, int]
    observation_stats: Dict[str, Dict[str, float]]
    temporal_consistency: float
    completeness_score: float
    quality_score: float


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Optional[QualityMetrics] = None


class DataValidator:
    """Validates processed replay data for quality and consistency."""
    
    # Expected observation shapes for validation
    EXPECTED_SHAPES = {
        'player_frame': (64, 64, 3),
        'global_view': (128, 128, 3),
        'game_state_minimal': (17,),
        'game_state_rich': (31,)
    }
    
    # Valid action range
    VALID_ACTIONS = set(range(6))  # Actions 0-5
    
    # Expected data types
    EXPECTED_DTYPES = {
        'player_frame': np.uint8,
        'global_view': np.uint8,
        'game_state': np.float32,
        'actions': np.int32
    }
    
    def validate_dataset(self, dataset_path: Path) -> ValidationResult:
        """
        Validate a complete dataset file.
        
        Args:
            dataset_path: Path to NPZ dataset file
            
        Returns:
            ValidationResult with validation status and metrics
        """
        errors = []
        warnings = []
        
        try:
            # Load dataset
            data = np.load(dataset_path, allow_pickle=True)
            
            # Check required keys
            required_keys = ['observations', 'actions', 'meta']
            for key in required_keys:
                if key not in data:
                    errors.append(f"Missing required key: {key}")
            
            if errors:
                return ValidationResult(False, errors, warnings)
            
            observations = data['observations'].item()
            actions = data['actions']
            meta = data['meta'].item()
            
            # Validate observations
            obs_errors, obs_warnings = self._validate_observations(observations)
            errors.extend(obs_errors)
            warnings.extend(obs_warnings)
            
            # Validate actions
            action_errors, action_warnings = self._validate_actions(actions)
            errors.extend(action_errors)
            warnings.extend(action_warnings)
            
            # Validate metadata
            meta_errors, meta_warnings = self._validate_metadata(meta)
            errors.extend(meta_errors)
            warnings.extend(meta_warnings)
            
            # Check consistency between components
            consistency_errors = self._validate_consistency(observations, actions, meta)
            errors.extend(consistency_errors)
            
            # Compute quality metrics
            metrics = self._compute_quality_metrics(observations, actions, meta)
            
            is_valid = len(errors) == 0
            return ValidationResult(is_valid, errors, warnings, metrics)
            
        except Exception as e:
            errors.append(f"Failed to load dataset: {e}")
            return ValidationResult(False, errors, warnings)
    
    def _validate_observations(self, observations: Dict[str, np.ndarray]) -> Tuple[List[str], List[str]]:
        """Validate observation data."""
        errors = []
        warnings = []
        
        # Check required observation keys
        required_keys = ['player_frame', 'global_view', 'game_state']
        for key in required_keys:
            if key not in observations:
                errors.append(f"Missing observation key: {key}")
        
        if errors:
            return errors, warnings
        
        # Validate shapes and dtypes
        for key, array in observations.items():
            if key == 'player_frame':
                expected_shape = self.EXPECTED_SHAPES['player_frame']
                if array.shape[1:] != expected_shape:
                    errors.append(f"Invalid {key} shape: {array.shape[1:]} != {expected_shape}")
                
                if array.dtype != self.EXPECTED_DTYPES['player_frame']:
                    errors.append(f"Invalid {key} dtype: {array.dtype} != {self.EXPECTED_DTYPES['player_frame']}")
                
                # Check value range for images
                if array.min() < 0 or array.max() > 255:
                    warnings.append(f"{key} values outside [0, 255] range")
            
            elif key == 'global_view':
                expected_shape = self.EXPECTED_SHAPES['global_view']
                if array.shape[1:] != expected_shape:
                    errors.append(f"Invalid {key} shape: {array.shape[1:]} != {expected_shape}")
                
                if array.dtype != self.EXPECTED_DTYPES['global_view']:
                    errors.append(f"Invalid {key} dtype: {array.dtype} != {self.EXPECTED_DTYPES['global_view']}")
                
                # Check value range for images
                if array.min() < 0 or array.max() > 255:
                    warnings.append(f"{key} values outside [0, 255] range")
            
            elif key == 'game_state':
                # Check if it's minimal or rich profile
                if array.shape[1] == 17:
                    expected_shape = self.EXPECTED_SHAPES['game_state_minimal']
                elif array.shape[1] == 31:
                    expected_shape = self.EXPECTED_SHAPES['game_state_rich']
                else:
                    errors.append(f"Invalid {key} shape: {array.shape[1]} not in [17, 31]")
                    continue
                
                if array.shape[1:] != expected_shape:
                    errors.append(f"Invalid {key} shape: {array.shape[1:]} != {expected_shape}")
                
                if array.dtype != self.EXPECTED_DTYPES['game_state']:
                    errors.append(f"Invalid {key} dtype: {array.dtype} != {self.EXPECTED_DTYPES['game_state']}")
                
                # Check for NaN or infinite values
                if np.any(np.isnan(array)) or np.any(np.isinf(array)):
                    errors.append(f"{key} contains NaN or infinite values")
                
                # Check reasonable value ranges (most should be normalized to [0, 1] or [-1, 1])
                if array.min() < -10 or array.max() > 10:
                    warnings.append(f"{key} values outside reasonable range [-10, 10]")
        
        return errors, warnings
    
    def _validate_actions(self, actions: np.ndarray) -> Tuple[List[str], List[str]]:
        """Validate action data."""
        errors = []
        warnings = []
        
        # Check dtype
        if actions.dtype != self.EXPECTED_DTYPES['actions']:
            errors.append(f"Invalid actions dtype: {actions.dtype} != {self.EXPECTED_DTYPES['actions']}")
        
        # Check action values are in valid range
        invalid_actions = set(actions) - self.VALID_ACTIONS
        if invalid_actions:
            errors.append(f"Invalid action values: {invalid_actions}")
        
        # Check for reasonable action distribution
        action_counts = np.bincount(actions, minlength=6)
        total_actions = len(actions)
        
        # Warn if any action is completely missing
        missing_actions = [i for i, count in enumerate(action_counts) if count == 0]
        if missing_actions:
            warnings.append(f"Actions never used: {missing_actions}")
        
        # Warn if action distribution is very skewed
        max_freq = action_counts.max() / total_actions
        if max_freq > 0.8:
            warnings.append(f"Action distribution highly skewed: max frequency {max_freq:.2f}")
        
        return errors, warnings
    
    def _validate_metadata(self, meta: Dict[str, np.ndarray]) -> Tuple[List[str], List[str]]:
        """Validate metadata."""
        errors = []
        warnings = []
        
        # Check required metadata keys
        required_keys = ['timestamp', 'level_id', 'frame_number', 'quality_score', 'session_id']
        for key in required_keys:
            if key not in meta:
                errors.append(f"Missing metadata key: {key}")
        
        if errors:
            return errors, warnings
        
        # Validate timestamps
        timestamps = meta['timestamp']
        if not np.all(timestamps[:-1] <= timestamps[1:]):
            warnings.append("Timestamps not monotonically increasing")
        
        # Validate frame numbers
        frame_numbers = meta['frame_number']
        if not np.all(frame_numbers >= 0):
            errors.append("Negative frame numbers found")
        
        # Validate quality scores
        quality_scores = meta['quality_score']
        if np.any(quality_scores < 0) or np.any(quality_scores > 1):
            warnings.append("Quality scores outside [0, 1] range")
        
        return errors, warnings
    
    def _validate_consistency(self, observations: Dict[str, np.ndarray], 
                            actions: np.ndarray, meta: Dict[str, np.ndarray]) -> List[str]:
        """Validate consistency between observations, actions, and metadata."""
        errors = []
        
        # Check that all arrays have the same length
        obs_lengths = [len(array) for array in observations.values()]
        action_length = len(actions)
        meta_lengths = [len(array) for array in meta.values()]
        
        all_lengths = obs_lengths + [action_length] + meta_lengths
        if len(set(all_lengths)) > 1:
            errors.append(f"Inconsistent array lengths: {set(all_lengths)}")
        
        return errors
    
    def _compute_quality_metrics(self, observations: Dict[str, np.ndarray],
                               actions: np.ndarray, meta: Dict[str, np.ndarray]) -> QualityMetrics:
        """Compute quality metrics for the dataset."""
        total_samples = len(actions)
        
        # Action distribution
        action_distribution = {i: int(count) for i, count in enumerate(np.bincount(actions, minlength=6))}
        
        # Observation statistics
        observation_stats = {}
        for key, array in observations.items():
            if key in ['player_frame', 'global_view']:
                # Image statistics
                observation_stats[key] = {
                    'mean': float(array.mean()),
                    'std': float(array.std()),
                    'min': float(array.min()),
                    'max': float(array.max())
                }
            elif key == 'game_state':
                # Game state statistics
                observation_stats[key] = {
                    'mean': float(array.mean()),
                    'std': float(array.std()),
                    'min': float(array.min()),
                    'max': float(array.max()),
                    'nan_count': int(np.sum(np.isnan(array))),
                    'inf_count': int(np.sum(np.isinf(array)))
                }
        
        # Temporal consistency (based on timestamps)
        timestamps = meta['timestamp']
        time_diffs = np.diff(timestamps)
        temporal_consistency = 1.0 - (np.std(time_diffs) / np.mean(time_diffs)) if len(time_diffs) > 0 else 1.0
        temporal_consistency = max(0.0, min(1.0, temporal_consistency))
        
        # Completeness score (no missing data)
        completeness_score = 1.0  # Assume complete if validation passed
        
        # Overall quality score (average of quality scores in metadata)
        quality_score = float(meta['quality_score'].mean())
        
        return QualityMetrics(
            total_samples=total_samples,
            valid_samples=total_samples,  # All samples are valid if we reach here
            invalid_samples=0,
            action_distribution=action_distribution,
            observation_stats=observation_stats,
            temporal_consistency=temporal_consistency,
            completeness_score=completeness_score,
            quality_score=quality_score
        )


class TrajectoryDeduplicator:
    """Removes duplicate trajectories from datasets."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize deduplicator.
        
        Args:
            similarity_threshold: Threshold for considering trajectories similar (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.trajectory_hashes = set()
        self.duplicate_count = 0
    
    def compute_trajectory_hash(self, actions: np.ndarray, meta: Dict[str, Any]) -> str:
        """
        Compute a hash for a trajectory based on actions and metadata.
        
        Args:
            actions: Action sequence
            meta: Trajectory metadata
            
        Returns:
            Hash string for the trajectory
        """
        # Create a signature from actions and key metadata
        signature_data = {
            'actions': actions.tolist(),
            'level_id': meta.get('level_id', ['unknown'])[0] if isinstance(meta.get('level_id'), np.ndarray) else meta.get('level_id', 'unknown'),
            'length': len(actions),
            'action_counts': np.bincount(actions, minlength=6).tolist()
        }
        
        # Convert to JSON string and hash
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()
    
    def compute_trajectory_similarity(self, actions1: np.ndarray, actions2: np.ndarray) -> float:
        """
        Compute similarity between two action sequences.
        
        Args:
            actions1: First action sequence
            actions2: Second action sequence
            
        Returns:
            Similarity score between 0 and 1
        """
        if len(actions1) != len(actions2):
            # Different lengths - compute similarity based on overlap
            min_len = min(len(actions1), len(actions2))
            max_len = max(len(actions1), len(actions2))
            
            # Compare the overlapping portion
            overlap_similarity = np.mean(actions1[:min_len] == actions2[:min_len])
            
            # Penalize for length difference
            length_penalty = min_len / max_len
            
            return overlap_similarity * length_penalty
        else:
            # Same length - direct comparison
            return np.mean(actions1 == actions2)
    
    def is_duplicate(self, actions: np.ndarray, meta: Dict[str, Any]) -> bool:
        """
        Check if a trajectory is a duplicate of a previously seen one.
        
        Args:
            actions: Action sequence
            meta: Trajectory metadata
            
        Returns:
            True if trajectory is considered a duplicate
        """
        trajectory_hash = self.compute_trajectory_hash(actions, meta)
        
        if trajectory_hash in self.trajectory_hashes:
            self.duplicate_count += 1
            return True
        
        self.trajectory_hashes.add(trajectory_hash)
        return False
    
    def deduplicate_dataset(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """
        Remove duplicates from a dataset file.
        
        Args:
            input_path: Input dataset file
            output_path: Output deduplicated dataset file
            
        Returns:
            Deduplication statistics
        """
        # Load dataset
        data = np.load(input_path, allow_pickle=True)
        observations = data['observations'].item()
        actions = data['actions']
        meta = data['meta'].item()
        
        # Track which samples to keep
        keep_indices = []
        original_count = len(actions)
        
        # Process each sample
        for i in range(len(actions)):
            # Extract metadata for this sample
            sample_meta = {key: value[i] for key, value in meta.items()}
            
            # Check if this trajectory is a duplicate
            if not self.is_duplicate(actions[i:i+1], sample_meta):
                keep_indices.append(i)
        
        # Filter data
        if keep_indices:
            filtered_observations = {}
            for key, value in observations.items():
                filtered_observations[key] = value[keep_indices]
            
            filtered_actions = actions[keep_indices]
            
            filtered_meta = {}
            for key, value in meta.items():
                filtered_meta[key] = value[keep_indices]
            
            # Save deduplicated dataset
            np.savez_compressed(
                output_path,
                observations=filtered_observations,
                actions=filtered_actions,
                meta=filtered_meta
            )
        
        # Return statistics
        kept_count = len(keep_indices)
        removed_count = original_count - kept_count
        
        return {
            'original_count': original_count,
            'kept_count': kept_count,
            'removed_count': removed_count,
            'removal_rate': removed_count / original_count if original_count > 0 else 0.0
        }


def generate_quality_report(dataset_paths: List[Path], output_path: Path):
    """Generate a comprehensive quality report for datasets."""
    validator = DataValidator()
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'datasets': {},
        'summary': {
            'total_datasets': len(dataset_paths),
            'valid_datasets': 0,
            'invalid_datasets': 0,
            'total_samples': 0,
            'total_errors': 0,
            'total_warnings': 0
        }
    }
    
    for dataset_path in dataset_paths:
        logger.info(f"Validating {dataset_path}")
        
        result = validator.validate_dataset(dataset_path)
        
        dataset_report = {
            'path': str(dataset_path),
            'is_valid': result.is_valid,
            'errors': result.errors,
            'warnings': result.warnings,
            'metrics': asdict(result.metrics) if result.metrics else None
        }
        
        report['datasets'][dataset_path.name] = dataset_report
        
        # Update summary
        if result.is_valid:
            report['summary']['valid_datasets'] += 1
        else:
            report['summary']['invalid_datasets'] += 1
        
        if result.metrics:
            report['summary']['total_samples'] += result.metrics.total_samples
        
        report['summary']['total_errors'] += len(result.errors)
        report['summary']['total_warnings'] += len(result.warnings)
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Quality report saved to {output_path}")
    return report


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data quality validation and deduplication tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate quality report
  python tools/data_quality.py --input datasets/processed --output quality_report.json
  
  # Deduplicate datasets
  python tools/data_quality.py --deduplicate datasets/processed --threshold 0.95
  
  # Validate single dataset
  python tools/data_quality.py --validate datasets/processed/batch_0000.npz
        """
    )
    
    parser.add_argument('--input', type=Path, help='Input directory with dataset files')
    parser.add_argument('--output', type=Path, help='Output path for quality report')
    parser.add_argument('--validate', type=Path, help='Validate single dataset file')
    parser.add_argument('--deduplicate', type=Path, help='Deduplicate datasets in directory')
    parser.add_argument('--threshold', type=float, default=0.95,
                       help='Similarity threshold for deduplication')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.validate:
        # Validate single dataset
        validator = DataValidator()
        result = validator.validate_dataset(args.validate)
        
        print(f"Validation Result for {args.validate}:")
        print(f"Valid: {result.is_valid}")
        
        if result.errors:
            print(f"Errors ({len(result.errors)}):")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.warnings:
            print(f"Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        if result.metrics:
            print("\nQuality Metrics:")
            print(f"  Total samples: {result.metrics.total_samples}")
            print(f"  Quality score: {result.metrics.quality_score:.3f}")
            print(f"  Temporal consistency: {result.metrics.temporal_consistency:.3f}")
            print(f"  Action distribution: {result.metrics.action_distribution}")
    
    elif args.input and args.output:
        # Generate quality report
        dataset_paths = list(args.input.glob('*.npz'))
        if not dataset_paths:
            logger.error(f"No NPZ files found in {args.input}")
            return
        
        report = generate_quality_report(dataset_paths, args.output)
        
        print("\nQuality Report Summary:")
        print(f"Total datasets: {report['summary']['total_datasets']}")
        print(f"Valid datasets: {report['summary']['valid_datasets']}")
        print(f"Invalid datasets: {report['summary']['invalid_datasets']}")
        print(f"Total samples: {report['summary']['total_samples']}")
        print(f"Total errors: {report['summary']['total_errors']}")
        print(f"Total warnings: {report['summary']['total_warnings']}")
    
    elif args.deduplicate:
        # Deduplicate datasets
        deduplicator = TrajectoryDeduplicator(args.threshold)
        dataset_paths = list(args.deduplicate.glob('*.npz'))
        
        if not dataset_paths:
            logger.error(f"No NPZ files found in {args.deduplicate}")
            return
        
        output_dir = args.deduplicate / 'deduplicated'
        output_dir.mkdir(exist_ok=True)
        
        total_stats = {
            'original_count': 0,
            'kept_count': 0,
            'removed_count': 0
        }
        
        for dataset_path in dataset_paths:
            output_path = output_dir / dataset_path.name
            stats = deduplicator.deduplicate_dataset(dataset_path, output_path)
            
            logger.info(f"Deduplicated {dataset_path.name}: "
                       f"{stats['removed_count']}/{stats['original_count']} removed "
                       f"({stats['removal_rate']:.1%})")
            
            for key in total_stats:
                total_stats[key] += stats[key]
        
        print("\nDeduplication Summary:")
        print(f"Original samples: {total_stats['original_count']}")
        print(f"Kept samples: {total_stats['kept_count']}")
        print(f"Removed samples: {total_stats['removed_count']}")
        if total_stats['original_count'] > 0:
            removal_rate = total_stats['removed_count'] / total_stats['original_count']
            print(f"Removal rate: {removal_rate:.1%}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
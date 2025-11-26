"""Unit tests for path prediction training losses."""

import unittest
import torch
import numpy as np

from npp_rl.path_prediction.training_losses import (
    chamfer_distance,
    waypoint_prediction_loss,
    confidence_calibration_loss,
    path_diversity_loss,
    compute_path_prediction_loss,
    PathPredictionLoss,
)


class TestChamferDistance(unittest.TestCase):
    """Test cases for Chamfer distance calculation."""
    
    def test_identical_point_sets(self):
        """Test Chamfer distance is zero for identical point sets."""
        points = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        
        dist = chamfer_distance(points, points)
        
        self.assertAlmostEqual(dist.item(), 0.0, places=5)
    
    def test_different_point_sets(self):
        """Test Chamfer distance is positive for different point sets."""
        pred_points = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        target_points = torch.tensor([[[10.0, 10.0], [20.0, 20.0]]])
        
        dist = chamfer_distance(pred_points, target_points)
        
        self.assertGreater(dist.item(), 0.0)
    
    def test_with_masks(self):
        """Test Chamfer distance with point masks."""
        # Padded sequences
        pred_points = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]])
        target_points = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]])
        
        # Masks indicate first two points are valid
        pred_mask = torch.tensor([[True, True, False]])
        target_mask = torch.tensor([[True, True, False]])
        
        dist = chamfer_distance(pred_points, target_points, pred_mask, target_mask)
        
        # Should be close to zero since valid points match
        self.assertLess(dist.item(), 1e-4)
    
    def test_batch_processing(self):
        """Test Chamfer distance works with batches."""
        batch_size = 4
        num_points = 5
        
        pred_points = torch.randn(batch_size, num_points, 2)
        target_points = torch.randn(batch_size, num_points, 2)
        
        dist = chamfer_distance(pred_points, target_points)
        
        # Should return scalar
        self.assertEqual(dist.shape, torch.Size([]))
        self.assertGreater(dist.item(), 0.0)


class TestWaypointPredictionLoss(unittest.TestCase):
    """Test cases for waypoint prediction loss."""
    
    def test_perfect_prediction(self):
        """Test loss is low when prediction matches expert."""
        batch_size = 2
        num_candidates = 4
        max_waypoints = 10
        
        # Expert waypoints
        expert = torch.randn(batch_size, max_waypoints, 2)
        expert_mask = torch.ones(batch_size, max_waypoints, dtype=torch.bool)
        
        # Predictions where first candidate matches expert
        predicted = torch.randn(batch_size, num_candidates, max_waypoints, 2)
        predicted[:, 0, :, :] = expert.unsqueeze(1)
        pred_mask = torch.ones(batch_size, num_candidates, max_waypoints, dtype=torch.bool)
        
        loss = waypoint_prediction_loss(
            predicted, expert, pred_mask, expert_mask, use_chamfer=True
        )
        
        # Loss should be very small since one candidate matches
        self.assertLess(loss.item(), 1e-4)
    
    def test_poor_prediction(self):
        """Test loss is high when prediction doesn't match expert."""
        batch_size = 2
        num_candidates = 4
        max_waypoints = 10
        
        # Expert waypoints
        expert = torch.randn(batch_size, max_waypoints, 2)
        expert_mask = torch.ones(batch_size, max_waypoints, dtype=torch.bool)
        
        # Very different predictions
        predicted = torch.randn(batch_size, num_candidates, max_waypoints, 2) + 100.0
        pred_mask = torch.ones(batch_size, num_candidates, max_waypoints, dtype=torch.bool)
        
        loss = waypoint_prediction_loss(
            predicted, expert, pred_mask, expert_mask, use_chamfer=True
        )
        
        # Loss should be high
        self.assertGreater(loss.item(), 1.0)


class TestConfidenceCalibrationLoss(unittest.TestCase):
    """Test cases for confidence calibration loss."""
    
    def test_correct_confidence_ranking(self):
        """Test loss when confidence correctly ranks best path."""
        batch_size = 2
        num_candidates = 4
        max_waypoints = 10
        
        # Expert waypoints
        expert = torch.randn(batch_size, max_waypoints, 2)
        expert_mask = torch.ones(batch_size, max_waypoints, dtype=torch.bool)
        
        # Predictions where first candidate matches expert
        predicted = torch.randn(batch_size, num_candidates, max_waypoints, 2)
        predicted[:, 0, :, :] = expert.unsqueeze(1)
        pred_mask = torch.ones(batch_size, num_candidates, max_waypoints, dtype=torch.bool)
        
        # Confidences with highest for first candidate
        confidences = torch.tensor([[10.0, 1.0, 1.0, 1.0], [10.0, 1.0, 1.0, 1.0]])
        
        loss = confidence_calibration_loss(
            confidences, predicted, expert, pred_mask, expert_mask
        )
        
        # Loss should be relatively low
        self.assertLess(loss.item(), 2.0)
    
    def test_incorrect_confidence_ranking(self):
        """Test loss when confidence incorrectly ranks paths."""
        batch_size = 2
        num_candidates = 4
        max_waypoints = 10
        
        # Expert waypoints
        expert = torch.randn(batch_size, max_waypoints, 2)
        expert_mask = torch.ones(batch_size, max_waypoints, dtype=torch.bool)
        
        # Predictions where first candidate matches expert
        predicted = torch.randn(batch_size, num_candidates, max_waypoints, 2)
        predicted[:, 0, :, :] = expert.unsqueeze(1)
        pred_mask = torch.ones(batch_size, num_candidates, max_waypoints, dtype=torch.bool)
        
        # Confidences with lowest for first candidate (incorrect)
        confidences = torch.tensor([[1.0, 10.0, 10.0, 10.0], [1.0, 10.0, 10.0, 10.0]])
        
        loss = confidence_calibration_loss(
            confidences, predicted, expert, pred_mask, expert_mask
        )
        
        # Loss should be higher due to incorrect ranking
        self.assertGreater(loss.item(), 1.0)


class TestPathDiversityLoss(unittest.TestCase):
    """Test cases for path diversity loss."""
    
    def test_diverse_paths(self):
        """Test loss is low when paths are diverse."""
        batch_size = 2
        num_candidates = 4
        max_waypoints = 10
        
        # Create diverse paths (far apart)
        predicted = torch.randn(batch_size, num_candidates, max_waypoints, 2)
        predicted[:, 0, :, :] *= 100  # Very different from others
        pred_mask = torch.ones(batch_size, num_candidates, max_waypoints, dtype=torch.bool)
        
        loss = path_diversity_loss(predicted, pred_mask, min_diversity=50.0)
        
        # Should be low or zero since paths are diverse
        self.assertLess(loss.item(), 1.0)
    
    def test_identical_paths(self):
        """Test loss is high when paths are identical."""
        batch_size = 2
        num_candidates = 4
        max_waypoints = 10
        
        # All paths are the same
        base_path = torch.randn(batch_size, max_waypoints, 2)
        predicted = base_path.unsqueeze(1).repeat(1, num_candidates, 1, 1)
        pred_mask = torch.ones(batch_size, num_candidates, max_waypoints, dtype=torch.bool)
        
        loss = path_diversity_loss(predicted, pred_mask, min_diversity=50.0)
        
        # Should be high since all paths are identical
        self.assertGreater(loss.item(), 10.0)
    
    def test_single_candidate_no_loss(self):
        """Test loss is zero when only one candidate path."""
        batch_size = 2
        num_candidates = 1
        max_waypoints = 10
        
        predicted = torch.randn(batch_size, num_candidates, max_waypoints, 2)
        pred_mask = torch.ones(batch_size, num_candidates, max_waypoints, dtype=torch.bool)
        
        loss = path_diversity_loss(predicted, pred_mask)
        
        # Should be zero (no pairs to compare)
        self.assertEqual(loss.item(), 0.0)


class TestComputePathPredictionLoss(unittest.TestCase):
    """Test cases for complete path prediction loss computation."""
    
    def test_loss_components(self):
        """Test that all loss components are computed."""
        batch_size = 2
        num_candidates = 4
        max_waypoints = 10
        
        predicted = torch.randn(batch_size, num_candidates, max_waypoints, 2)
        confidences = torch.randn(batch_size, num_candidates)
        expert = torch.randn(batch_size, max_waypoints, 2)
        pred_mask = torch.ones(batch_size, num_candidates, max_waypoints, dtype=torch.bool)
        expert_mask = torch.ones(batch_size, max_waypoints, dtype=torch.bool)
        
        loss_result = compute_path_prediction_loss(
            predicted, confidences, expert, pred_mask, expert_mask
        )
        
        # Check all components exist
        self.assertIsInstance(loss_result, PathPredictionLoss)
        self.assertIsInstance(loss_result.total_loss, torch.Tensor)
        self.assertIsInstance(loss_result.waypoint_loss, torch.Tensor)
        self.assertIsInstance(loss_result.confidence_loss, torch.Tensor)
        self.assertIsInstance(loss_result.diversity_loss, torch.Tensor)
        
        # All should be non-negative
        self.assertGreaterEqual(loss_result.total_loss.item(), 0.0)
        self.assertGreaterEqual(loss_result.waypoint_loss.item(), 0.0)
        self.assertGreaterEqual(loss_result.confidence_loss.item(), 0.0)
        self.assertGreaterEqual(loss_result.diversity_loss.item(), 0.0)
    
    def test_loss_weights(self):
        """Test that loss weights are applied correctly."""
        batch_size = 2
        num_candidates = 4
        max_waypoints = 10
        
        predicted = torch.randn(batch_size, num_candidates, max_waypoints, 2)
        confidences = torch.randn(batch_size, num_candidates)
        expert = torch.randn(batch_size, max_waypoints, 2)
        pred_mask = torch.ones(batch_size, num_candidates, max_waypoints, dtype=torch.bool)
        expert_mask = torch.ones(batch_size, max_waypoints, dtype=torch.bool)
        
        # Compute with default weights
        loss1 = compute_path_prediction_loss(
            predicted, confidences, expert, pred_mask, expert_mask,
            waypoint_loss_weight=1.0,
            confidence_loss_weight=0.5,
            diversity_loss_weight=0.3,
        )
        
        # Compute with different weights
        loss2 = compute_path_prediction_loss(
            predicted, confidences, expert, pred_mask, expert_mask,
            waypoint_loss_weight=2.0,
            confidence_loss_weight=1.0,
            diversity_loss_weight=0.6,
        )
        
        # Total losses should be different
        self.assertNotAlmostEqual(
            loss1.total_loss.item(), 
            loss2.total_loss.item(),
            places=3
        )
    
    def test_loss_backward(self):
        """Test that loss can be used for backpropagation."""
        batch_size = 2
        num_candidates = 4
        max_waypoints = 10
        
        predicted = torch.randn(batch_size, num_candidates, max_waypoints, 2, requires_grad=True)
        confidences = torch.randn(batch_size, num_candidates, requires_grad=True)
        expert = torch.randn(batch_size, max_waypoints, 2)
        pred_mask = torch.ones(batch_size, num_candidates, max_waypoints, dtype=torch.bool)
        expert_mask = torch.ones(batch_size, max_waypoints, dtype=torch.bool)
        
        loss_result = compute_path_prediction_loss(
            predicted, confidences, expert, pred_mask, expert_mask
        )
        
        # Should be able to compute gradients
        loss_result.total_loss.backward()
        
        self.assertIsNotNone(predicted.grad)
        self.assertIsNotNone(confidences.grad)


if __name__ == "__main__":
    unittest.main()


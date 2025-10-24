# Utility modules

from .performance_monitor import (
    PerformanceMonitor,
    TimingContext,
    time_function
)

try:
    from .s3_uploader import (
        S3Uploader,
        create_s3_uploader
    )
except ImportError:
    # boto3 not installed - S3 functionality unavailable
    S3Uploader = None
    create_s3_uploader = None

from .logging_utils import (
    setup_experiment_logging,
    save_experiment_config,
    load_experiment_config,
    TensorBoardManager,
    log_training_step,
    log_evaluation,
    create_experiment_summary,
    setup_comparison_logging
)

try:
    from .video_recorder import (
        VideoRecorder,
        create_video_recorder
    )
except ImportError:
    # imageio not installed - video recording unavailable
    VideoRecorder = None
    create_video_recorder = None

from .frame_stack_logging import (
    log_frame_stack_config,
    visualize_stacked_frames,
    log_stacked_observations,
    log_state_stack_statistics
)

__all__ = [
    'PerformanceMonitor',
    'TimingContext', 
    'time_function',
    'S3Uploader',
    'create_s3_uploader',
    'setup_experiment_logging',
    'save_experiment_config',
    'load_experiment_config',
    'TensorBoardManager',
    'log_training_step',
    'log_evaluation',
    'create_experiment_summary',
    'setup_comparison_logging',
    'VideoRecorder',
    'create_video_recorder',
    'log_frame_stack_config',
    'visualize_stacked_frames',
    'log_stacked_observations',
    'log_state_stack_statistics'
]
"""S3 artifact uploader for training experiments.

Handles uploading checkpoints, logs, and evaluation results to AWS S3
with manifest tracking and incremental sync support.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None


logger = logging.getLogger(__name__)


class S3Uploader:
    """Manages uploading training artifacts to S3."""
    
    def __init__(
        self, 
        bucket: str, 
        prefix: str, 
        experiment_name: str,
        dry_run: bool = False
    ):
        """Initialize S3 uploader.
        
        Args:
            bucket: S3 bucket name
            prefix: Base prefix for all uploads
            experiment_name: Unique experiment identifier
            dry_run: If True, log operations without uploading
        """
        if not BOTO3_AVAILABLE and not dry_run:
            raise ImportError(
                "boto3 is required for S3 uploads. "
                "Install with: pip install boto3"
            )
        
        self.bucket = bucket
        self.base_prefix = f"{prefix.rstrip('/')}/{experiment_name}"
        self.experiment_name = experiment_name
        self.dry_run = dry_run
        self.uploaded_files: List[Dict[str, Any]] = []
        
        if not dry_run:
            try:
                self.s3 = boto3.client('s3')
                # Test credentials
                self.s3.head_bucket(Bucket=bucket)
                logger.info(f"S3 uploader initialized: s3://{bucket}/{self.base_prefix}")
            except NoCredentialsError:
                raise ValueError(
                    "AWS credentials not found. Set AWS_ACCESS_KEY_ID and "
                    "AWS_SECRET_ACCESS_KEY environment variables or configure "
                    "AWS CLI with 'aws configure'"
                )
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                if error_code == '404':
                    raise ValueError(f"S3 bucket '{bucket}' does not exist")
                elif error_code == '403':
                    raise ValueError(
                        f"Access denied to bucket '{bucket}'. "
                        "Check IAM permissions"
                    )
                else:
                    raise
        else:
            self.s3 = None
            logger.info(f"S3 uploader in dry-run mode (no actual uploads)")
    
    def upload_file(
        self,
        local_path: str,
        s3_key: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Upload a single file to S3.
        
        Args:
            local_path: Path to local file
            s3_key: S3 key (relative to base prefix). If None, uses filename
            metadata: Optional metadata dict to attach to S3 object
            
        Returns:
            True if upload succeeded, False otherwise
        """
        local_path = Path(local_path)
        
        if not local_path.exists():
            logger.warning(f"File not found, skipping: {local_path}")
            return False
        
        if not local_path.is_file():
            logger.warning(f"Not a file, skipping: {local_path}")
            return False
        
        # Determine S3 key
        if s3_key is None:
            s3_key = local_path.name
        
        full_s3_key = f"{self.base_prefix}/{s3_key}"
        
        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would upload: {local_path} -> "
                f"s3://{self.bucket}/{full_s3_key}"
            )
            self.uploaded_files.append({
                'local_path': str(local_path),
                's3_key': full_s3_key,
                'timestamp': datetime.now().isoformat(),
                'dry_run': True
            })
            return True
        
        try:
            extra_args = {}
            if metadata:
                # Convert all metadata values to strings
                extra_args['Metadata'] = {
                    k: str(v) for k, v in metadata.items()
                }
            
            self.s3.upload_file(
                str(local_path), 
                self.bucket, 
                full_s3_key,
                ExtraArgs=extra_args if extra_args else None
            )
            
            self.uploaded_files.append({
                'local_path': str(local_path),
                's3_key': full_s3_key,
                'timestamp': datetime.now().isoformat(),
                'size_bytes': local_path.stat().st_size
            })
            
            logger.info(
                f"Uploaded: {local_path.name} -> "
                f"s3://{self.bucket}/{full_s3_key}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False
    
    def upload_directory(
        self,
        local_dir: str,
        s3_prefix: str,
        pattern: str = '*',
        recursive: bool = True
    ) -> int:
        """Upload all files in a directory matching pattern.
        
        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix (relative to base prefix)
            pattern: Glob pattern for file matching
            recursive: If True, search recursively
            
        Returns:
            Number of files successfully uploaded
        """
        local_path = Path(local_dir)
        
        if not local_path.exists() or not local_path.is_dir():
            logger.warning(f"Directory not found: {local_dir}")
            return 0
        
        upload_count = 0
        
        if recursive:
            files = local_path.rglob(pattern)
        else:
            files = local_path.glob(pattern)
        
        for file_path in files:
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{relative_path}"
                if self.upload_file(str(file_path), s3_key):
                    upload_count += 1
        
        logger.info(f"Uploaded {upload_count} files from {local_dir}")
        return upload_count
    
    def sync_tensorboard_logs(
        self, 
        tensorboard_dir: str, 
        s3_prefix: str
    ) -> int:
        """Sync TensorBoard event files to S3.
        
        Args:
            tensorboard_dir: Local TensorBoard log directory
            s3_prefix: S3 prefix for logs
            
        Returns:
            Number of files uploaded
        """
        return self.upload_directory(
            tensorboard_dir,
            s3_prefix,
            pattern='events.out.tfevents.*',
            recursive=True
        )
    
    def save_manifest(self, output_path: str) -> None:
        """Save manifest of uploaded files.
        
        Args:
            output_path: Local path to save manifest JSON
        """
        manifest = {
            'experiment_name': self.experiment_name,
            'bucket': self.bucket,
            'base_prefix': self.base_prefix,
            'dry_run': self.dry_run,
            'uploaded_files': self.uploaded_files,
            'total_files': len(self.uploaded_files),
            'total_size_bytes': sum(
                f.get('size_bytes', 0) for f in self.uploaded_files
            ),
            'created_at': datetime.now().isoformat()
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Saved manifest to {output_path}")
        
        # Upload manifest itself
        if not self.dry_run:
            self.upload_file(str(output_path), 's3_manifest.json')
    
    def get_s3_url(self, s3_key: str) -> str:
        """Get full S3 URL for a key.
        
        Args:
            s3_key: S3 key (relative to base prefix)
            
        Returns:
            Full S3 URL
        """
        full_key = f"{self.base_prefix}/{s3_key}"
        return f"s3://{self.bucket}/{full_key}"


def create_s3_uploader(
    bucket: Optional[str],
    prefix: str,
    experiment_name: str,
    dry_run: bool = False
) -> Optional[S3Uploader]:
    """Create S3 uploader with error handling.
    
    Args:
        bucket: S3 bucket name (None to disable S3)
        prefix: Base S3 prefix
        experiment_name: Experiment identifier
        dry_run: Enable dry-run mode
        
    Returns:
        S3Uploader instance or None if disabled
    """
    if bucket is None:
        logger.info("S3 upload disabled (no bucket specified)")
        return None
    
    try:
        uploader = S3Uploader(bucket, prefix, experiment_name, dry_run=dry_run)
        return uploader
    except Exception as e:
        logger.error(f"Failed to initialize S3 uploader: {e}")
        logger.warning("Continuing without S3 upload")
        return None

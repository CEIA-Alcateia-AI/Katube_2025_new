"""
Sox Audio Normalizer
Final normalization using Sox for audio quality standardization
"""
import os
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from .naming_utils import extract_base_name, generate_standard_name

logger = logging.getLogger(__name__)

class SoxNormalizer:
    """
    Final audio normalizer using Sox for quality standardization.
    Normalizes gain and applies final audio processing.
    """
    
    def __init__(self, 
                 target_sample_rate: int = 48000,
                 target_format: str = "flac",
                 target_channels: int = 1,
                 normalize_gain: bool = True):
        """
        Initialize Sox normalizer.
        
        Args:
            target_sample_rate: Target sample rate (default: 48000 Hz)
            target_format: Target audio format (default: flac)
            target_channels: Target number of channels (default: 1 = mono)
            normalize_gain: Whether to normalize gain (default: True)
        """
        self.target_sample_rate = target_sample_rate
        self.target_format = target_format
        self.target_channels = target_channels
        self.normalize_gain = normalize_gain
        
        # Check if Sox is available
        self._check_sox()
        
    def _check_sox(self):
        """Check if Sox is available in the system."""
        # Try to find Sox executable in common locations
        sox_paths = [
            'sox',  # Try PATH first
            r'C:\Program Files\Chris Bagwell\SoX\sox.exe',
            r'C:\Program Files (x86)\Chris Bagwell\SoX\sox.exe',
            # WinGet installation path pattern
            os.path.join(os.getenv('LOCALAPPDATA', ''), 'Microsoft', 'WinGet', 'Packages', 'ChrisBagwell.SoX_Microsoft.Winget.Source_8wekyb3d8bbwe', 'sox-14.4.2', 'sox.exe'),
            # Alternative WinGet path
            os.path.expanduser(r'~\AppData\Local\Microsoft\WinGet\Packages\ChrisBagwell.SoX_Microsoft.Winget.Source_8wekyb3d8bbwe\sox-14.4.2\sox.exe')
        ]
        
        sox_found = False
        sox_executable = None
        
        for sox_path in sox_paths:
            try:
                result = subprocess.run(
                    [sox_path, '--version'], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                if result.returncode == 0:
                    sox_found = True
                    sox_executable = sox_path
                    logger.info(f"✅ Sox found at: {sox_path}")
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                continue
        
        if not sox_found:
            raise RuntimeError("❌ Sox not found. Please install Sox: https://sox.sourceforge.net/")
            
        self.sox_executable = sox_executable
    
    def normalize_audio(self, input_path: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Normalize audio file using Sox.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for normalized output (if None, generates automatically)
            
        Returns:
            Dictionary with normalization results
        """
        try:
            # Generate output path if not provided
            if output_path is None:
                # Use standardized naming
                base_name = extract_base_name(input_path)
                standard_name = generate_standard_name(base_name, "sox_normalized")
                output_path = input_path.parent / f"{standard_name}.{self.target_format}"
            
            logger.info(f"🔄 Normalizing audio with Sox: {input_path.name}")
            logger.info(f"   Target: {self.target_format}, {self.target_sample_rate}Hz, {self.target_channels} channel(s)")
            
            # Build Sox command
            cmd = [
                self.sox_executable,
                str(input_path),                    # Input file
                '-r', str(self.target_sample_rate), # Sample rate
                '-c', str(self.target_channels),    # Channels (mono)
                str(output_path)                    # Output file
            ]
            
            # Add gain normalization if enabled
            if self.normalize_gain:
                cmd.insert(-1, '--norm')  # Insert before output path
            
            # Execute Sox command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                # Get file info
                input_size = input_path.stat().st_size
                output_size = output_path.stat().st_size
                
                logger.info(f"✅ Audio normalized successfully with Sox")
                logger.info(f"   Input: {input_size / (1024*1024):.1f} MB")
                logger.info(f"   Output: {output_size / (1024*1024):.1f} MB")
                logger.info(f"   Saved to: {output_path}")
                
                return {
                    'success': True,
                    'input_path': str(input_path),
                    'output_path': str(output_path),
                    'input_size_mb': input_size / (1024*1024),
                    'output_size_mb': output_size / (1024*1024),
                    'sample_rate': self.target_sample_rate,
                    'channels': self.target_channels,
                    'format': self.target_format,
                    'normalize_gain': self.normalize_gain
                }
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(f"❌ Sox normalization failed: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'input_path': str(input_path)
                }
                
        except subprocess.TimeoutExpired:
            error_msg = "Sox normalization timeout (5 minutes)"
            logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'input_path': str(input_path)
            }
        except Exception as e:
            error_msg = f"Sox normalization error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'input_path': str(input_path)
            }
    
    def normalize_audio_batch(self, input_paths: List[Path], output_dir: Path) -> Dict[str, Any]:
        """
        Normalize multiple audio files using Sox.
        
        Args:
            input_paths: List of input audio file paths
            output_dir: Directory to save normalized files
            
        Returns:
            Dictionary with batch normalization results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'successful': [],
            'failed': [],
            'total_processed': len(input_paths),
            'success_count': 0,
            'failure_count': 0
        }
        
        logger.info(f"🔄 Starting batch normalization with Sox: {len(input_paths)} files")
        
        for i, input_path in enumerate(input_paths):
            try:
                # Generate output path
                base_name = extract_base_name(input_path)
                standard_name = generate_standard_name(base_name, "sox_normalized", i+1)
                output_path = output_dir / f"{standard_name}.{self.target_format}"
                
                # Normalize single file
                result = self.normalize_audio(input_path, output_path)
                
                if result['success']:
                    results['successful'].append(result)
                    results['success_count'] += 1
                    logger.info(f"✅ {i+1}/{len(input_paths)}: {input_path.name} -> {output_path.name}")
                else:
                    results['failed'].append({
                        'input_path': str(input_path),
                        'error': result['error']
                    })
                    results['failure_count'] += 1
                    logger.error(f"❌ {i+1}/{len(input_paths)}: {input_path.name} - {result['error']}")
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"📈 Sox normalization progress: {i + 1}/{len(input_paths)} files")
                    
            except Exception as e:
                error_msg = f"Batch processing error: {str(e)}"
                results['failed'].append({
                    'input_path': str(input_path),
                    'error': error_msg
                })
                results['failure_count'] += 1
                logger.error(f"❌ {i+1}/{len(input_paths)}: {input_path.name} - {error_msg}")
        
        logger.info(f"🎯 Sox batch normalization complete:")
        logger.info(f"   ✅ Successful: {results['success_count']}")
        logger.info(f"   ❌ Failed: {results['failure_count']}")
        logger.info(f"   📁 Output directory: {output_dir}")
        
        return results

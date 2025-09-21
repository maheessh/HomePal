#!/usr/bin/env python3
"""
PyTorch to NCNN Model Converter

This module provides utilities to convert PyTorch models to NCNN format for optimized inference.
NCNN is a high-performance neural network inference framework optimized for mobile platforms.

Requirements:
- torch
- torchvision
- onnx
- onnxsim
- ncnn (optional, for testing converted models)

Usage:
    from packages.pytorch_to_ncnn_converter import PyTorchToNCNNConverter
    
    converter = PyTorchToNCNNConverter()
    converter.convert_model(
        pytorch_model_path="model.pt",
        output_dir="ncnn_model",
        input_shape=(1, 3, 320, 320),
        model_name="my_model"
    )
"""

import os
import shutil
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import torch
import torch.onnx
from ultralytics import YOLO


class PyTorchToNCNNConverter:
    """
    A comprehensive converter for PyTorch models to NCNN format.
    
    This converter handles the full pipeline:
    1. PyTorch -> ONNX
    2. ONNX optimization
    3. ONNX -> NCNN (requires external tools)
    4. Model testing and validation
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the converter.
        
        Args:
            verbose: Whether to print detailed conversion information
        """
        self.verbose = verbose
        self.temp_dir = None
        
    def _log(self, message: str):
        """Print log message if verbose is enabled."""
        if self.verbose:
            print(f"[Converter] {message}")
    
    def _check_dependencies(self) -> bool:
        """
        Check if required dependencies are available.
        
        Returns:
            bool: True if all dependencies are available
        """
        try:
            import onnx
            import onnxsim
            self._log("✓ ONNX and ONNXSIM dependencies found")
            return True
        except ImportError as e:
            self._log(f"✗ Missing dependency: {e}")
            self._log("Install with: pip install onnx onnxsim")
            return False
    
    def _create_dummy_input(self, input_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Create a dummy input tensor for model tracing.
        
        Args:
            input_shape: Shape of the input tensor (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Dummy input tensor
        """
        return torch.randn(input_shape, dtype=torch.float32)
    
    def _export_to_onnx(self, 
                        model: torch.nn.Module, 
                        dummy_input: torch.Tensor,
                        onnx_path: str,
                        opset_version: int = 11) -> bool:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            dummy_input: Dummy input tensor for tracing
            onnx_path: Path to save the ONNX model
            opset_version: ONNX opset version
            
        Returns:
            bool: True if export was successful
        """
        try:
            self._log(f"Exporting model to ONNX (opset {opset_version})...")
            
            # Set model to evaluation mode
            model.eval()
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            self._log(f"✓ ONNX model saved to: {onnx_path}")
            return True
            
        except Exception as e:
            self._log(f"✗ ONNX export failed: {e}")
            return False
    
    def _optimize_onnx(self, onnx_path: str, optimized_path: str) -> bool:
        """
        Optimize ONNX model using onnxsim.
        
        Args:
            onnx_path: Path to the original ONNX model
            optimized_path: Path to save the optimized ONNX model
            
        Returns:
            bool: True if optimization was successful
        """
        try:
            import onnx
            import onnxsim
            
            self._log("Optimizing ONNX model...")
            
            # Load the ONNX model
            model = onnx.load(onnx_path)
            
            # Simplify the model
            model_simp, check = onnxsim.simplify(model)
            
            if check:
                onnx.save(model_simp, optimized_path)
                self._log(f"✓ Optimized ONNX model saved to: {optimized_path}")
                return True
            else:
                self._log("✗ ONNX optimization failed - model check failed")
                return False
                
        except Exception as e:
            self._log(f"✗ ONNX optimization failed: {e}")
            return False
    
    def _convert_onnx_to_ncnn(self, onnx_path: str, output_dir: str) -> bool:
        """
        Convert ONNX model to NCNN format using external tools.
        
        Note: This requires the NCNN tools to be installed and available in PATH.
        
        Args:
            onnx_path: Path to the ONNX model
            output_dir: Directory to save NCNN files
            
        Returns:
            bool: True if conversion was successful
        """
        try:
            self._log("Converting ONNX to NCNN...")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Paths for NCNN files
            param_path = os.path.join(output_dir, "model.ncnn.param")
            bin_path = os.path.join(output_dir, "model.ncnn.bin")
            
            # Try to find onnx2ncnn tool
            onnx2ncnn_cmd = self._find_onnx2ncnn()
            
            if not onnx2ncnn_cmd:
                self._log("✗ onnx2ncnn tool not found in PATH")
                self._log("Please install NCNN tools and add them to PATH")
                self._log("Download from: https://github.com/Tencent/ncnn/releases")
                return False
            
            # Run conversion
            cmd = [onnx2ncnn_cmd, onnx_path, param_path, bin_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log(f"✓ NCNN model saved to: {output_dir}")
                self._log(f"  - Param file: {param_path}")
                self._log(f"  - Bin file: {bin_path}")
                return True
            else:
                self._log(f"✗ NCNN conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            self._log(f"✗ NCNN conversion failed: {e}")
            return False
    
    def _find_onnx2ncnn(self) -> Optional[str]:
        """
        Find the onnx2ncnn tool in the system PATH.
        
        Returns:
            Optional[str]: Path to onnx2ncnn tool, or None if not found
        """
        import shutil
        return shutil.which("onnx2ncnn")
    
    def _create_metadata(self, 
                        output_dir: str, 
                        model_name: str,
                        input_shape: Tuple[int, ...],
                        model_info: Dict) -> bool:
        """
        Create metadata file for the NCNN model.
        
        Args:
            output_dir: Directory containing the NCNN model
            model_name: Name of the model
            input_shape: Input tensor shape
            model_info: Additional model information
            
        Returns:
            bool: True if metadata creation was successful
        """
        try:
            metadata = {
                'name': model_name,
                'description': f'NCNN converted model from PyTorch',
                'version': '1.0.0',
                'author': 'PyTorchToNCNNConverter',
                'date': str(torch.utils.data.get_worker_info()),
                'input_shape': list(input_shape),
                'format': 'ncnn',
                'framework': 'pytorch',
                'task': model_info.get('task', 'unknown'),
                'classes': model_info.get('classes', []),
                'original_model': model_info.get('original_path', 'unknown')
            }
            
            metadata_path = os.path.join(output_dir, "metadata.yaml")
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
            
            self._log(f"✓ Metadata saved to: {metadata_path}")
            return True
            
        except Exception as e:
            self._log(f"✗ Metadata creation failed: {e}")
            return False
    
    def _create_test_script(self, output_dir: str, input_shape: Tuple[int, ...]) -> bool:
        """
        Create a test script for the NCNN model.
        
        Args:
            output_dir: Directory containing the NCNN model
            input_shape: Input tensor shape
            
        Returns:
            bool: True if test script creation was successful
        """
        try:
            test_script = f'''#!/usr/bin/env python3
"""
Test script for NCNN model inference.
Generated by PyTorchToNCNNConverter.
"""

import numpy as np
import ncnn
import torch

def test_inference():
    """Test the NCNN model with dummy input."""
    torch.manual_seed(0)
    
    # Create dummy input
    dummy_input = torch.randn({input_shape}, dtype=torch.float32)
    
    output = []
    
    try:
        with ncnn.Net() as net:
            # Load NCNN model
            net.load_param("model.ncnn.param")
            net.load_model("model.ncnn.bin")
            
            with net.create_extractor() as ex:
                # Input the dummy data
                ex.input("input", ncnn.Mat(dummy_input.squeeze(0).numpy()).clone())
                
                # Extract output
                _, out = ex.extract("output")
                output.append(torch.from_numpy(np.array(out)).unsqueeze(0))
        
        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)
            
    except Exception as e:
        print(f"Error during inference: {{e}}")
        return None

if __name__ == "__main__":
    result = test_inference()
    if result is not None:
        print("✓ NCNN model inference successful!")
        print(f"Output shape: {{result.shape}}")
        print(f"Output sample: {{result.flatten()[:5]}}")
    else:
        print("✗ NCNN model inference failed!")
'''
            
            test_script_path = os.path.join(output_dir, "model_ncnn.py")
            with open(test_script_path, 'w') as f:
                f.write(test_script)
            
            # Make script executable
            os.chmod(test_script_path, 0o755)
            
            self._log(f"✓ Test script saved to: {test_script_path}")
            return True
            
        except Exception as e:
            self._log(f"✗ Test script creation failed: {e}")
            return False
    
    def convert_yolo_model(self, 
                          yolo_model_path: str,
                          output_dir: str,
                          input_size: int = 320,
                          model_name: Optional[str] = None) -> bool:
        """
        Convert a YOLO model to NCNN format.
        
        Args:
            yolo_model_path: Path to the YOLO model (.pt file)
            output_dir: Directory to save the NCNN model
            input_size: Input image size for the model
            model_name: Name for the converted model (defaults to filename)
            
        Returns:
            bool: True if conversion was successful
        """
        try:
            self._log(f"Converting YOLO model: {yolo_model_path}")
            
            # Load YOLO model
            model = YOLO(yolo_model_path)
            pytorch_model = model.model
            
            # Set model name if not provided
            if model_name is None:
                model_name = Path(yolo_model_path).stem
            
            # Create dummy input
            input_shape = (1, 3, input_size, input_size)
            dummy_input = self._create_dummy_input(input_shape)
            
            # Model info for metadata
            model_info = {
                'task': 'detection',
                'classes': list(model.names.values()) if hasattr(model, 'names') else [],
                'original_path': yolo_model_path
            }
            
            # Perform conversion
            return self._convert_model(
                model=pytorch_model,
                dummy_input=dummy_input,
                output_dir=output_dir,
                model_name=model_name,
                model_info=model_info
            )
            
        except Exception as e:
            self._log(f"✗ YOLO model conversion failed: {e}")
            return False
    
    def convert_model(self,
                     model: Union[torch.nn.Module, str],
                     output_dir: str,
                     input_shape: Tuple[int, ...],
                     model_name: str,
                     model_info: Optional[Dict] = None,
                     opset_version: int = 11) -> bool:
        """
        Convert a PyTorch model to NCNN format.
        
        Args:
            model: PyTorch model or path to .pt file
            output_dir: Directory to save the NCNN model
            input_shape: Shape of the input tensor (batch_size, channels, height, width)
            model_name: Name for the converted model
            model_info: Additional model information for metadata
            opset_version: ONNX opset version for export
            
        Returns:
            bool: True if conversion was successful
        """
        try:
            # Check dependencies
            if not self._check_dependencies():
                return False
            
            # Load model if path is provided
            if isinstance(model, str):
                self._log(f"Loading PyTorch model from: {model}")
                model = torch.load(model, map_location='cpu')
                if isinstance(model, dict) and 'model' in model:
                    model = model['model']
            
            # Set model to evaluation mode
            model.eval()
            
            # Create dummy input
            dummy_input = self._create_dummy_input(input_shape)
            
            # Create temporary directory for intermediate files
            with tempfile.TemporaryDirectory() as temp_dir:
                self.temp_dir = temp_dir
                
                # Step 1: Export to ONNX
                onnx_path = os.path.join(temp_dir, "model.onnx")
                if not self._export_to_onnx(model, dummy_input, onnx_path, opset_version):
                    return False
                
                # Step 2: Optimize ONNX
                optimized_onnx_path = os.path.join(temp_dir, "model_optimized.onnx")
                if not self._optimize_onnx(onnx_path, optimized_onnx_path):
                    # If optimization fails, use original ONNX
                    optimized_onnx_path = onnx_path
                    self._log("⚠ Using original ONNX model (optimization failed)")
                
                # Step 3: Convert ONNX to NCNN
                if not self._convert_onnx_to_ncnn(optimized_onnx_path, output_dir):
                    return False
                
                # Step 4: Create metadata
                if model_info is None:
                    model_info = {}
                if not self._create_metadata(output_dir, model_name, input_shape, model_info):
                    self._log("⚠ Metadata creation failed, continuing...")
                
                # Step 5: Create test script
                if not self._create_test_script(output_dir, input_shape):
                    self._log("⚠ Test script creation failed, continuing...")
            
            self._log(f"✓ Model conversion completed successfully!")
            self._log(f"  Output directory: {output_dir}")
            self._log(f"  Model name: {model_name}")
            self._log(f"  Input shape: {input_shape}")
            
            return True
            
        except Exception as e:
            self._log(f"✗ Model conversion failed: {e}")
            return False
        
        finally:
            self.temp_dir = None


def convert_yolo_to_ncnn(yolo_path: str, 
                        output_dir: str, 
                        input_size: int = 320,
                        verbose: bool = True) -> bool:
    """
    Convenience function to convert a YOLO model to NCNN format.
    
    Args:
        yolo_path: Path to the YOLO model (.pt file)
        output_dir: Directory to save the NCNN model
        input_size: Input image size for the model
        verbose: Whether to print conversion progress
        
    Returns:
        bool: True if conversion was successful
        
    Example:
        success = convert_yolo_to_ncnn(
            yolo_path="yolov8n.pt",
            output_dir="yolov8n_ncnn",
            input_size=320
        )
    """
    converter = PyTorchToNCNNConverter(verbose=verbose)
    return converter.convert_yolo_model(yolo_path, output_dir, input_size)


def convert_pytorch_to_ncnn(model_path: str,
                           output_dir: str,
                           input_shape: Tuple[int, ...],
                           model_name: str,
                           verbose: bool = True) -> bool:
    """
    Convenience function to convert a PyTorch model to NCNN format.
    
    Args:
        model_path: Path to the PyTorch model (.pt file)
        output_dir: Directory to save the NCNN model
        input_shape: Shape of the input tensor
        model_name: Name for the converted model
        verbose: Whether to print conversion progress
        
    Returns:
        bool: True if conversion was successful
        
    Example:
        success = convert_pytorch_to_ncnn(
            model_path="my_model.pt",
            output_dir="my_model_ncnn",
            input_shape=(1, 3, 224, 224),
            model_name="my_model"
        )
    """
    converter = PyTorchToNCNNConverter(verbose=verbose)
    return converter.convert_model(
        model=model_path,
        output_dir=output_dir,
        input_shape=input_shape,
        model_name=model_name
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PyTorch models to NCNN format")
    parser.add_argument("--model", required=True, help="Path to PyTorch model (.pt file)")
    parser.add_argument("--output", required=True, help="Output directory for NCNN model")
    parser.add_argument("--input-shape", nargs=4, type=int, default=[1, 3, 320, 320],
                       help="Input shape as batch_size channels height width")
    parser.add_argument("--name", help="Model name (defaults to filename)")
    parser.add_argument("--input-size", type=int, default=320,
                       help="Input size for YOLO models")
    parser.add_argument("--yolo", action="store_true", help="Convert YOLO model")
    
    args = parser.parse_args()
    
    converter = PyTorchToNCNNConverter(verbose=True)
    
    if args.yolo:
        # Convert YOLO model
        success = converter.convert_yolo_model(
            yolo_model_path=args.model,
            output_dir=args.output,
            input_size=args.input_size,
            model_name=args.name
        )
    else:
        # Convert regular PyTorch model
        success = converter.convert_model(
            model=args.model,
            output_dir=args.output,
            input_shape=tuple(args.input_shape),
            model_name=args.name or Path(args.model).stem
        )
    
    if success:
        print("✓ Conversion completed successfully!")
    else:
        print("✗ Conversion failed!")
        exit(1)

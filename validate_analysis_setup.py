"""
Validation Script for Experiment Analysis Framework

This script checks that all required components are in place and provides
diagnostic information for troubleshooting the refactored analysis framework.

Usage:
    python validate_analysis_setup.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

def check_file_exists(filepath: Path) -> Tuple[bool, str]:
    """Check if a file exists and return status."""
    if filepath.exists():
        size_kb = filepath.stat().st_size / 1024
        return True, f"✓ Found ({size_kb:.1f} KB)"
    
    return False, "✗ Missing"


def check_python_packages() -> List[Dict[str, str]]:
    """Check if required Python packages are installed."""
    required_packages = [
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn"),
        ("joblib", "Joblib")
    ]
    
    results = []
    
    for module_name, display_name in required_packages:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            results.append({
                "package": display_name,
                "status": "✓",
                "version": version
            })
        except ImportError:
            results.append({
                "package": display_name,
                "status": "✗",
                "version": "Not installed"
            })
    
    return results


def check_project_structure() -> List[Dict[str, str]]:
    """Check if key project files exist."""
    root = Path.cwd()
    
    files_to_check = [
        ("experiment_analysis_framework_refactored.ipynb", "Refactored Notebook"),
        ("experiment_analysis_utils.py", "Analysis Utilities"),
        ("james_stein_constraint.py", "James-Stein Constraint"),
        ("advanced_neural_network.py", "Advanced Neural Network"),
        ("data_processing.py", "Data Processing"),
        ("weight_constraints.py", "Weight Constraints"),
        ("ml_utils.py", "ML Utilities"),
        ("requirements.txt", "Requirements"),
        ("ml_config/model_config.json", "Model Config"),
        ("ml_config/training_config.json", "Training Config"),
    ]
    
    results = []
    
    for filepath, description in files_to_check:
        full_path = root / filepath
        exists, status = check_file_exists(full_path)
        
        results.append({
            "file": description,
            "path": filepath,
            "status": status
        })
    
    return results


def check_training_artifacts() -> List[Dict[str, str]]:
    """Check if training artifacts exist."""
    root = Path.cwd()
    output_dir = root / "training_output"
    
    artifacts_to_check = [
        ("loss_history.csv", "Loss History"),
        ("training_results.csv", "Training Results"),
        ("configuration_log.csv", "Configuration Log"),
    ]
    
    results = []
    
    for filename, description in artifacts_to_check:
        full_path = output_dir / filename
        exists, status = check_file_exists(full_path)
        
        results.append({
            "artifact": description,
            "path": str(full_path.relative_to(root)) if exists else filename,
            "status": status
        })
    
    # Check for weight checkpoints
    weight_dir = root / "saved_weights"
    
    if weight_dir.exists():
        checkpoints = list(weight_dir.glob("model_weights_epoch_*.weights.h5"))
        checkpoint_status = f"✓ Found {len(checkpoints)} checkpoint(s)"
    else:
        checkpoint_status = "✗ Directory missing"
    
    results.append({
        "artifact": "Weight Checkpoints",
        "path": "saved_weights/",
        "status": checkpoint_status
    })
    
    return results


def main():
    """Run all validation checks and display results."""
    print("=" * 80)
    print("Experiment Analysis Framework - Setup Validation")
    print("=" * 80)
    print()
    
    # Check Python packages
    print("1. Python Package Check")
    print("-" * 80)
    
    package_results = check_python_packages()
    
    for result in package_results:
        print(f"  {result['status']} {result['package']:<20} {result['version']}")
    
    missing_packages = [r for r in package_results if r["status"] == "✗"]
    
    if missing_packages:
        print("\n  ⚠️  Missing packages detected. Install with:")
        print("     pip install -r requirements.txt")
    
    print()
    
    # Check project structure
    print("2. Project Structure Check")
    print("-" * 80)
    
    structure_results = check_project_structure()
    
    for result in structure_results:
        print(f"  {result['status']:<20} {result['file']:<30} {result['path']}")
    
    missing_files = [r for r in structure_results if "✗" in r["status"]]
    
    if missing_files:
        print("\n  ⚠️  Missing core files detected.")
        print("     Ensure you're in the project root directory.")
    
    print()
    
    # Check training artifacts
    print("3. Training Artifacts Check")
    print("-" * 80)
    
    artifact_results = check_training_artifacts()
    
    for result in artifact_results:
        print(f"  {result['status']:<30} {result['artifact']:<25} {result['path']}")
    
    missing_artifacts = [r for r in artifact_results if "✗" in r["status"]]
    
    if missing_artifacts:
        print("\n  ℹ️  Training artifacts missing. Generate them with:")
        print("     python main.py")
        print("\n  Note: The notebook can run without artifacts but will have limited data.")
    
    print()
    
    # Overall status
    print("=" * 80)
    print("Overall Status")
    print("=" * 80)
    
    all_checks_passed = (
        len(missing_packages) == 0 and
        len(missing_files) == 0
    )
    
    if all_checks_passed:
        print("✅ Setup validation PASSED")
        print()
        print("Next steps:")
        print("  1. Run training: python main.py")
        print("  2. Open notebook: jupyter notebook experiment_analysis_framework_refactored.ipynb")
        print("  3. Execute all cells to generate analysis")
    else:
        print("⚠️  Setup validation FAILED")
        print()
        print("Please address the issues above before running the analysis notebook.")
    
    print()
    
    # Additional info
    print("Additional Information")
    print("-" * 80)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Working directory: {Path.cwd()}")
    print()
    print("For more information, see:")
    print("  • README.md - Full documentation")
    print("  • REFACTORING_SUMMARY.md - Delivery details")
    print()


if __name__ == "__main__":
    main()

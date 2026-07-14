#!/usr/bin/env python
"""
Main script to run all experiments
"""
import argparse
import subprocess
import sys
from pathlib import Path

def setup_environment():
    """Setup environment and download data"""
    print("Setting up environment...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])
    
    # Download data
    print("Downloading data...")
    subprocess.run([
        sys.executable, "scripts/download_data.py"
    ])

def run_experiments(all_iterations=True):
    """Run the experiments"""
    from experiments.experiment_runner import AURAExperimentRunner as ExperimentRunner
    
    runner = ExperimentRunner()
    
    if all_iterations:
        runner.run_all()
    else:
        # Run specific iterations
        for iteration in runner.config['iterations']:
            runner.run_iteration(iteration['name'])

def generate_presentation():
    """Generate presentation materials"""
    print("Generating presentation materials...")
    subprocess.run([
        sys.executable, "scripts/generate_presentation.py"
    ])

def main():
    parser = argparse.ArgumentParser(description="Run music search experiments")
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--setup', action='store_true', help='Setup environment')
    parser.add_argument('--eval-only', action='store_true', help='Evaluate only (no indexing)')
    parser.add_argument('--generate-presentation', action='store_true', help='Generate presentation')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_environment()
    
    if args.all:
        run_experiments(all_iterations=True)
    elif args.eval_only:
        # Load existing results and evaluate
        from experiments.experiment_runner import ExperimentRunner
        runner = ExperimentRunner()
        for iteration in runner.config['iterations']:
            # Load cached index and evaluate
            pass
    elif args.generate_presentation:
        generate_presentation()
    else:
        # Default: run all
        run_experiments(all_iterations=True)

if __name__ == "__main__":
    main()

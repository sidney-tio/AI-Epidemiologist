import os
import re
import argparse
from ai_scientist.perform_writeup import compile_latex


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Latex for Experiments")
    parser.add_argument("--results_dir", type=str, help="results directory")
    args = parser.parse_args()
    results_dir = os.path.join("results", args.results_dir)
    pattern = r"\d{8}_\d{6}_(.+)"

    for folder in os.listdir(results_dir):
        if os.path.isdir(os.path.join(results_dir, folder)):
            print(f"{folder}")
            match = re.search(pattern, folder)
            if match:
                print("compiling...")
                idea = match.group(1)
                cwd = os.path.join(results_dir, folder, "latex")
                pdf_file = os.path.join(results_dir, folder, f"{idea}.pdf")
                compile_latex(cwd, pdf_file)

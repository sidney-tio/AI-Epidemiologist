import os.path as osp
import os
from datetime import datetime
import sys
import re
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from typing import Optional, List
from ai_scientist.perform_writeup import perform_writeup


def print_time():
    """Print current timestamp"""
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def get_matching_folders(base_dir: str, pattern: str) -> List[tuple[str, str]]:
    """
    Find folders in the base directory that match the given pattern.
    Returns list of tuples containing (folder_path, extracted_name).
    """
    pattern = re.compile(pattern)
    matching_folders = []

    # List all items in the base directory
    for item in os.listdir(base_dir):
        full_path = osp.join(base_dir, item)
        if osp.isdir(full_path):
            match = pattern.match(item)
            if match:
                # Extract the name from the pattern match
                extracted_name = match.group(1)
                matching_folders.append((full_path, extracted_name))

    return sorted(matching_folders)


def process_folder(
    folder_path: str,
    folder_name: str,
    model: str = "deepseek-coder-v2-0724",
    writeup: str = "latex",
    log_file: str = None,
    io=None,
    client=None,
    client_model=None,
) -> bool:
    """
    Process a single folder containing idea content.

    Args:
        folder_path: Full path to the folder
        folder_name: Extracted name of the folder
        model: Model identifier string
        writeup: Format for writeup ('latex' supported)
        log_file: Optional path to log file
        io: IO interface object
        client: Client object
        client_model: Client model object

    Returns:
        bool: True if processing successful, False otherwise
    """
    # Setup logging if specified
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if log_file:
        try:
            log = open(log_file, "w")
            sys.stdout = log
            sys.stderr = log
        except Exception as e:
            print(f"Failed to open log file: {e}")
            return False

    try:
        if writeup == "latex":
            # Setup file paths
            notes = osp.join(folder_path, "analysis.txt")
            writeup_file = osp.join(folder_path, "latex", "template.tex")
            fnames = [writeup_file, notes]

            # Initialize model
            if model == "deepseek-coder-v2-0724":
                main_model = Model("deepseek/deepseek-coder")
            elif model == "llama3.1-405b":
                main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
            else:
                main_model = Model(model)

            # Initialize coder
            coder = Coder.create(
                main_model=main_model,
                fnames=fnames,
                io=io,
                stream=False,
                use_git=False,
                edit_format="diff",
            )

            # Read idea content from analysis.txt
            try:
                with open(notes, "r") as f:
                    idea = f.read()
            except Exception as e:
                print(f"Failed to read analysis.txt: {e}")
                return False

            # Perform writeup
            try:
                perform_writeup(idea, folder_path, coder, client, client_model)
            except Exception as e:
                print(f"Failed to perform writeup: {e}")
                return False
            print("Done writeup")
        else:
            raise ValueError(f"Writeup format {writeup} not supported.")

        print_time()
        return True

    except Exception as e:
        print(f"Failed to evaluate idea {folder_name}: {str(e)}")
        return False

    finally:
        print("FINISHED IDEA")
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()


def main():
    """
    Main function to process all matching folders in the results directory.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Process and evaluate ideas using AI models"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Base directory containing result folders",
    )
    parser.add_argument(
        "--model", default="gpt-4o-2024-05-13", help="Model identifier string"
    )
    parser.add_argument("--writeup", default="latex", help="Format for writeup")
    parser.add_argument("--log-dir", help="Directory for log files")

    args = parser.parse_args()

    # Pattern for matching folder names
    pattern = r"\d{8}_\d{6}_(.+)"

    # Find matching folders
    matching_folders = get_matching_folders(args.results_dir, pattern)

    if not matching_folders:
        print(f"No matching folders found in {args.results_dir}")
        return False

    # Process each matching folder
    success = True
    for folder_path, folder_name in matching_folders:
        print(f"\nProcessing folder: {folder_path}")

        # Setup log file if log directory is specified
        log_file = None
        if args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)
            log_file = osp.join(args.log_dir, f"{folder_name}.log")

        # Note: io, client, and client_model would need to be initialized based on your specific needs
        folder_success = process_folder(
            folder_path=folder_path,
            folder_name=folder_name,
            model=args.model,
            writeup=args.writeup,
            log_file=log_file,
            io=None,  # Initialize as needed
            client=None,  # Initialize as needed
            client_model=None,  # Initialize as needed
        )

        success = success and folder_success

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

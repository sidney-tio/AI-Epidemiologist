import os
import shutil
import subprocess
from subprocess import TimeoutExpired
import sys
from typing import Dict, Any


MAX_ITERS = 4
MAX_INTERVIEWS = 5
MAX_STDERR_OUTPUT = 1500

interviewer_prompt = """Your goal is to conduct interviews for the following policy: {title}.
The interview objective is as follows: {objective}.
You are given a total of up to {max_interviews} interviews to complete. You do not need to use all {max_interviews}.
DO NOT MAKE EDITS TO experiment.py.
First, plan the list of questions that you will want to ask each interviewee and write it to questions.json. In the first question, make sure to present the overall study to contextualize the interview.
Next, plan the list of personas you would like to interview. Consider different backgrounds, expertise levels, and perspectives that would provide valuable insights and all-rounded views for the research topic.
After you create each persona, we will run an interview with the created persona.
You can then implement the next persona on your list by writing to persona.txt."""


def run_interview(
    folder_name: str, interview_num: int, timeout: int = 3600
) -> tuple[int, str]:
    """Run a single interview with the specified persona."""
    cwd = os.path.abspath(folder_name)

    # Copy the interview script for reference
    shutil.copy(
        os.path.join(folder_name, "questions.json"),
        os.path.join(folder_name, f"question_{interview_num}.json"),
    )
    shutil.copy(
        os.path.join(folder_name, "persona.txt"),
        os.path.join(folder_name, f"persona_{interview_num}.txt"),
    )
    # Launch command
    command = [
        "python",
        "experiment.py",
        f"--out_dir=interview_{interview_num}",
    ]

    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(
                f"Interview {interview_num} failed with return code {result.returncode}"
            )
            if os.path.exists(os.path.join(cwd, f"interview_{interview_num}")):
                shutil.rmtree(os.path.join(cwd, f"interview_{interview_num}"))
            stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            next_prompt = f"Interview failed with the following error {stderr_output}"
        else:
            # Read interview results
            with open(
                os.path.join(
                    cwd, f"interview_{interview_num}", "interview_summary.txt"
                ),
                "r",
            ) as f:
                interview_text = f.read()

            next_prompt = f"""Interview {interview_num} completed successfully. Here is the conversation summary:
{interview_text}

Please review the interview and provide:
1. A brief analysis of the key insights gained
2. Whether this persona provided the perspectives we were looking for
3. What type of persona we should interview next

Decide if you need to change the questions or add more questions.

Someone else will be using `analysis.txt` to perform a writeup on this in the future.
Please include *all* relevant information about Interview {interview_num}, including the persona description and key findings into analysis.txt. Append new information only, and do not edit or erase anything that is already in the file.

Then, implement the next persona on your list.
We will then run the command `python experiment.py --out_dir=interview_{interview_num + 1}'.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
If you are finished with interviews, respond with 'ALL_COMPLETED'."""

        return result.returncode, next_prompt

    except TimeoutExpired:
        print(f"Interview {interview_num} timed out after {timeout} seconds")
        if os.path.exists(os.path.join(cwd, f"interview_{interview_num}")):
            shutil.rmtree(os.path.join(cwd, f"interview_{interview_num}"))
        next_prompt = f"Interview timed out after {timeout} seconds"
        return 1, next_prompt


def conduct_interviews(research: Dict[str, str], folder_name: str, aider: Any) -> bool:
    """Main function to conduct all interviews and generate analysis."""
    current_iter = 0
    interview_num = 1

    # Initialize with the research objectives
    next_prompt = interviewer_prompt.format(
        title=research["Title"],
        objective=research["Experiment"],
        max_interviews=MAX_INTERVIEWS,
    )

    # Conduct interviews
    while interview_num < MAX_INTERVIEWS + 1:
        if current_iter >= MAX_ITERS:
            print("Max iterations reached")
            break

        aider_out = aider.run(next_prompt)
        print(aider_out)

        if "ALL_COMPLETED" in aider_out:
            break

        return_code, next_prompt = run_interview(folder_name, interview_num)

        if return_code == 0:
            interview_num += 1
            current_iter = 0
        current_iter += 1

    if current_iter >= MAX_ITERS:
        print("Not all interviews completed.")
        return False

    # Run final analysis
    current_iter = 0
    next_prompt = """
Great job with the interviews! You have seen a summary of all the interviews conducted.
Please enrich `analysis.txt` by performing cross-interview analysis.

You should:
1. Identify common themes and patterns
2. Extract key insights
3. Identify unexpected insights that emerged
4. How well the different personas complemented each other
5. Based on the insights from the interviews, propose any modifications, if any, to the original proposal for a policymaker to consider and implement.

Do not edit or remove any of the prior discussions that are in analysis.txt, and only append to the file.
Someone else will be using this analysis to write a report in the future.
"""
    aider.run(next_prompt)

    return True


def main():
    # Example usage
    research = {
        "Title": "Impact of AI on Healthcare Decision Making",
        "Objective": "Investigate how different healthcare professionals view and use AI in their decision-making processes",
    }

    folder_name = "interview_project"
    aider = None  # Replace with actual Aider instance

    success = conduct_interviews(research, folder_name, aider)
    if success:
        print("Interview project completed successfully")
    else:
        print("Interview project did not complete all planned interviews")


if __name__ == "__main__":
    main()

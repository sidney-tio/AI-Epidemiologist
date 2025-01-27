import os
import argparse
import json
import openai
import time
from typing import List, Dict
import os


class InterviewSimulator:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def load_questions(self, file_path: str) -> List[str]:
        """Load questions from JSON file."""
        with open(file_path, "r") as f:
            return json.load(f)

    def load_persona(self, file_path: str) -> str:
        """Load persona description from text file."""
        with open(file_path, "r") as f:
            return f.read().strip()

    def get_interviewer_response(
        self, questions: List[str], persona: str, conversation_history: List[Dict]
    ) -> str:
        """Generate interviewer's next question or comment."""
        system_prompt = f"""You are an expert interviewer that will be interviewing {persona}. You have these base questions to work with:
{json.dumps(questions, indent=2)}

However, you're free to:
1. Ask follow-up questions based on the interviewee's responses
2. Share relevant insights or observations
3. Skip questions if they've been indirectly answered
4. Adjust the order of questions based on conversation flow

Your goal is to conduct an engaging, natural interview while getting detailed responses.
Keep your questions and comments concise and focused.
You may end the interview once you feel that the interview has achieved its objective by returning INTERVIEW_COMPLETE
"""

        messages = [{"role": "system", "content": system_prompt}]

        # Convert conversation history to interviewer's perspective
        for msg in conversation_history:
            if msg["role"] == "assistant" and msg.get("speaker") == "interviewer":
                # Previous interviewer messages become assistant messages
                messages.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "user" and msg.get("speaker") == "interviewee":
                # Interviewee messages become user messages
                messages.append({"role": "user", "content": msg["content"]})

        response = self.client.chat.completions.create(
            model="gpt-4", messages=messages, temperature=0.7, max_tokens=300
        )

        return response.choices[0].message.content

    def get_interviewee_response(
        self, persona: str, conversation_history: List[Dict]
    ) -> str:
        """Generate interviewee's response."""
        system_prompt = f"""You are being interviewed. Here is your persona:

{persona}

Respond naturally and in character, drawing from the provided persona details.
Be consistent with your character's background, expertise, and personality traits.
Provide detailed, thoughtful answers while maintaining authenticity."""

        messages = [{"role": "system", "content": system_prompt}]

        # Convert conversation history to interviewee's perspective
        for msg in conversation_history:
            if msg["role"] == "assistant" and msg.get("speaker") == "interviewer":
                # Interviewer messages become user messages
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "user" and msg.get("speaker") == "interviewee":
                # Previous interviewee messages become assistant messages
                messages.append({"role": "assistant", "content": msg["content"]})

        response = self.client.chat.completions.create(
            model="gpt-4", messages=messages, temperature=0.7, max_tokens=1000
        )

        return response.choices[0].message.content

    def generate_interview_summary(
        self, conversation_history: List[Dict], persona: str
    ) -> str:
        """Generate a comprehensive summary of the interview."""
        # Prepare the interview text in a readable format
        interview_text = "Interview Transcript:\n\n"
        for message in conversation_history:
            speaker = message.get("speaker", "")
            if speaker == "interviewer":
                interview_text += f"Interviewer: {message['content']}\n"
            elif speaker == "interviewee":
                interview_text += f"Interviewee: {message['content']}\n"
            interview_text += "\n"

        system_prompt = """You are an expert interview analyst. Your task is to provide a comprehensive summary of the interview.
Include:
1. Key themes and topics discussed
2. Main insights and notable quotes
3. Interviewee's expertise and perspective
4. Notable patterns in responses
5. Areas where the interviewee showed particular enthusiasm or concern
6. Any unique or unexpected viewpoints
7. Overall tone and engagement level of the conversation

Organize your summary in clear sections with headers."""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Here is the interviewee's persona:

{persona}

And here is the full interview:

{interview_text}

Please provide a comprehensive summary.""",
            },
        ]

        response = self.client.chat.completions.create(
            model="gpt-4", messages=messages, temperature=0.7, max_tokens=2000
        )

        return response.choices[0].message.content

    def save_interview(self, conversation: List[Dict], output_file: str, persona: str):
        """Save the interview transcript and summary."""
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the transcript
        with open(output_file, "w") as f:
            f.write("Interview Transcript\n")
            f.write("==================\n\n")

            for message in conversation:
                speaker = message.get("speaker", "")
                if speaker == "interviewer":
                    f.write(f"\nInterviewer: {message['content']}\n")
                elif speaker == "interviewee":
                    f.write(f"\nInterviewee: {message['content']}\n")

        # Generate and save the summary
        summary = self.generate_interview_summary(conversation, persona)
        summary_file = os.path.join(output_dir, "interview_summary.txt")
        with open(summary_file, "w") as f:
            f.write("Interview Summary\n")
            f.write("================\n\n")
            f.write(summary)

    def conduct_interview(
        self,
        questions_file: str,
        persona_file: str,
        output_file: str,
        max_questions: int,
    ):
        """Main method to conduct the interview."""
        questions = self.load_questions(questions_file)
        persona = self.load_persona(persona_file)
        conversation_history = []
        question_count = 0

        print("Starting interview simulation...")

        while question_count < max_questions:
            # Get interviewer's question
            interviewer_response = self.get_interviewer_response(
                questions, persona, conversation_history
            )
            conversation_history.append(
                {
                    "role": "assistant",
                    "content": interviewer_response,
                    "speaker": "interviewer",
                }
            )
            print(f"\nInterviewer: {interviewer_response}")

            if "INTERVIEW_COMPLETE" in interviewer_response:
                break
            # Add delay for rate limiting
            time.sleep(1)

            # Get interviewee's response
            interviewee_response = self.get_interviewee_response(
                persona, conversation_history
            )
            conversation_history.append(
                {
                    "role": "user",
                    "content": interviewee_response,
                    "speaker": "interviewee",
                }
            )
            print(f"\nInterviewee: {interviewee_response}")

            question_count += 1

            # Add delay for rate limiting
            time.sleep(1)

        # Save the interview
        self.save_interview(conversation_history, output_file, persona)
        print(f"\nInterview completed and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Initialize simulator
    simulator = InterviewSimulator(api_key)

    # Run interview
    simulator.conduct_interview(
        questions_file="questions.json",
        persona_file="persona.txt",
        output_file=os.path.join(out_dir, "interview.txt"),
        max_questions=10,  # Adjust this number as needed
    )


if __name__ == "__main__":
    main()

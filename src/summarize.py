import argparse
import sys

from mistral_client import MistralClient

DEPTHS = ("light", "standard", "deep")

SYSTEM_PROMPT = (
    "You are an expert analyst. You write structured markdown summaries "
    "of video transcripts. You always answer in the same language as the "
    "transcript you are given."
)

DEPTH_INSTRUCTIONS = {
    "light": """Produce:
- A "## Key takeaways" section with 3 to 5 concise bullet points.
- A "## Overview" section with a single paragraph.""",
    "standard": """Produce:
- A "## Key takeaways" section with 5 to 8 bullet points.
- A "## Main arguments" section describing the main arguments or strategies discussed.
- A "## Notable quotes and data points" section.""",
    "deep": """Produce:
- A "## Key takeaways" section with 8 to 12 bullet points.
- A "## Argument breakdown" section with a comprehensive breakdown and supporting evidence.
- A "## Notable quotes" section, including timestamps when the transcript contains them.
- A "## Data points and statistics" section.
- A "## Counterarguments and caveats" section raised by the speaker.""",
}


def summarize_transcript(
    client: MistralClient,
    transcript: str,
    title: str,
    channel: str,
    language: str,
    depth: str = "standard",
) -> str:
    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {DEPTHS}, got: {depth}")
    prompt = f"""Here is the full transcript of a YouTube video.

Title: {title}
Channel: {channel}
Transcript language: {language}

{DEPTH_INSTRUCTIONS[depth]}

Write the summary in the same language as the transcript (language code: {language}).
Start with the title as a level-1 markdown heading, then the channel name.

Transcript:
---
{transcript}
---

Return only the markdown summary, nothing else."""
    return client.chat(prompt, system=SYSTEM_PROMPT)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a structured summary of a video transcript with Mistral AI"
    )
    parser.add_argument("transcript_file", help="Path to the transcript .txt file")
    parser.add_argument("--title", required=True, help="Video title")
    parser.add_argument("--channel", required=True, help="Channel name")
    parser.add_argument("--language", required=True, help="Transcript language code")
    parser.add_argument(
        "--depth",
        choices=DEPTHS,
        default="standard",
        help="Summary detail level (default: standard)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .md file (default: stdout)",
    )
    args = parser.parse_args()

    with open(args.transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read()
    if not transcript.strip():
        print("Error: transcript file is empty")
        sys.exit(1)

    client = MistralClient()
    summary = summarize_transcript(
        client, transcript, args.title, args.channel, args.language, args.depth
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summary + "\n")
        print(f"SUMMARY_FILE:{args.output}")
    else:
        print(summary)


if __name__ == "__main__":
    main()

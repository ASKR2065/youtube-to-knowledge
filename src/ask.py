import argparse
import sys
from pathlib import Path

from mistral_client import MistralClient

SYSTEM_PROMPT = """You are a video analyst. You answer questions about video transcripts.
Rules:
- Only use information contained in the transcript provided.
- Quote with timestamps when the transcript contains them.
- For data points, give the exact numbers stated in the transcript.
- If the transcript doesn't contain the answer, say so clearly. Never fabricate.
- Answer in the language of the question."""


def find_transcript(video_ref: str) -> tuple[Path, Path | None]:
    content_dir = Path(__file__).resolve().parent.parent / "vault" / "content"
    for channel_dir in sorted(content_dir.iterdir()):
        if not channel_dir.is_dir():
            continue
        raw_dir = channel_dir / "raw"
        if not raw_dir.exists():
            continue
        for transcript in sorted(raw_dir.glob("transcript_*.txt")):
            if video_ref.lower() in transcript.stem.lower() or video_ref in channel_dir.name.lower():
                enriched_candidates = list(raw_dir.glob("transcript_enriched_*.txt")) or list(
                    raw_dir.glob("*enriched*.txt")
                )
                enriched = enriched_candidates[0] if enriched_candidates else None
                return transcript, enriched
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question about a processed video's transcript (Mistral AI)"
    )
    parser.add_argument(
        "video",
        help="Video ID, or part of the channel name / transcript file name",
    )
    parser.add_argument("question", help="The question to answer")
    args = parser.parse_args()

    transcript, enriched = find_transcript(args.video)
    if not transcript:
        print(f"Error: no transcript found for '{args.video}'. Process the video first.")
        sys.exit(1)

    source = enriched or transcript
    print(f"Using transcript: {source}")
    with open(source, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    client = MistralClient()
    prompt = f"""Transcript:
---
{transcript_text}
---

Question: {args.question}"""
    answer = client.chat(prompt, system=SYSTEM_PROMPT, temperature=0.2)
    print(f"\n{answer}")


if __name__ == "__main__":
    main()

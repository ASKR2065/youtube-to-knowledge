import argparse
import json
import re
import sys

from mistral_client import MistralClient

DEPTHS = ("light", "standard", "deep")

TRIPLET_COUNTS = {"light": "8-12", "standard": "15-20", "deep": "25-35"}

DEPTH_INSTRUCTIONS = {
    "light": "Extract only primary entities and their direct relationships.",
    "standard": "Extract people, companies, tools, concepts, and their relationships.",
    "deep": """Extract:
- Primary relationships (people, companies, tools, concepts)
- Causal chains ("X leads to Y" as separate triplets)
- Temporal relationships ("X preceded Y", "X replaced Y")
- Attributed claims (attribute to the speaker when known)
- Quantitative relationships ("X grew by 300%")""",
}

SYSTEM_PROMPT = """You extract knowledge-graph triplets from video transcripts.
You return ONLY a valid JSON array, with no markdown fences and no commentary.
Each element is an object: {"subject": str, "predicate": str, "object": str}.
Rules:
- Use full canonical names for entities (e.g. "Mistral AI", not "Mistral").
- Predicates must be specific verbs: "founded", "recommends", "competes_with", "grew_by". Never vague predicates like "is related to" or "involves".
- Each triplet must be verifiable from the transcript. No inferences."""


def parse_triplets(raw: str) -> list[dict]:
    text = raw.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON array found in model output")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("model output is not a JSON array")
    triplets = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Warning: triplet {i} is not an object, skipping")
            continue
        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()
        if not subject or not predicate or not obj:
            print(f"Warning: triplet {i} missing a field, skipping")
            continue
        triplets.append({"subject": subject, "predicate": predicate, "object": obj})
    return triplets


def extract_triplets(
    client: MistralClient,
    transcript: str,
    channel: str,
    depth: str = "standard",
    existing_entities: list[str] | None = None,
) -> list[dict]:
    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {DEPTHS}, got: {depth}")
    entities_hint = ""
    if existing_entities:
        shown = ", ".join(existing_entities[:200])
        entities_hint = (
            "\nEntities already present in the knowledge graph: "
            f"[{shown}]. Reuse these exact names when the concept is the same."
        )
    prompt = f"""Extract {TRIPLET_COUNTS[depth]} knowledge-graph triplets from this transcript.

Channel / main speaker: {channel}
{DEPTH_INSTRUCTIONS[depth]}{entities_hint}

Transcript:
---
{transcript}
---

Return only the JSON array."""
    raw = client.chat(prompt, system=SYSTEM_PROMPT, temperature=0.2)
    return parse_triplets(raw)


def main():
    parser = argparse.ArgumentParser(
        description="Extract knowledge-graph triplets from a transcript with Mistral AI"
    )
    parser.add_argument("transcript_file", help="Path to the transcript .txt file")
    parser.add_argument("--channel", required=True, help="Channel / speaker name")
    parser.add_argument(
        "--depth",
        choices=DEPTHS,
        default="standard",
        help="Number of triplets to target (default: standard)",
    )
    parser.add_argument(
        "--graph",
        default=None,
        help="Existing graph.json to reuse entity names from (optional)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .json file (default: stdout)",
    )
    args = parser.parse_args()

    with open(args.transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read()
    if not transcript.strip():
        print("Error: transcript file is empty")
        sys.exit(1)

    existing_entities = None
    if args.graph:
        try:
            with open(args.graph, "r", encoding="utf-8") as f:
                data = json.load(f)
            existing_entities = [str(node.get("id", "")) for node in data.get("nodes", [])]
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: could not read graph {args.graph}: {e}")

    client = MistralClient()
    triplets = extract_triplets(
        client, transcript, args.channel, args.depth, existing_entities
    )
    payload = json.dumps(triplets, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        print(f"TRIPLETS_FILE:{args.output}")
        print(f"TRIPLETS_COUNT:{len(triplets)}")
    else:
        print(payload)


if __name__ == "__main__":
    main()

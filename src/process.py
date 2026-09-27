import argparse
import os
import re
import subprocess
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)


def run_script(script: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, os.path.join(SRC_DIR, script), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def parse_env(output: str) -> dict:
    env = {}
    for line in output.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            if key in ("CHANNEL_DIR", "RAW_FILE", "SOURCE_LANG", "TITLE", "ENRICHED_FILE"):
                env[key] = value.strip()
    return env


def extract_video_id(url: str) -> str | None:
    match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None


def expand_playlist(url: str) -> list[str]:
    print(f"Expanding playlist: {url}")
    result = run_script("playlist_extractor.py", [url])
    if result.returncode != 0:
        print(result.stderr.strip() or "Error: playlist expansion failed")
        sys.exit(1)
    urls = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    print(f"Found {len(urls)} video(s) in playlist")
    return urls


def is_duplicate(video_id: str) -> bool:
    db_path = os.path.join(REPO_ROOT, "vault", "processed_videos.md")
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            return f"`{video_id}`" in f.read()
    return False


def transcribe(url: str, engine: str) -> tuple[dict, str]:
    print("Trying fast transcription (YouTube captions)...")
    result = run_script("transcribe.py", [url])
    env = parse_env(result.stdout)
    if result.returncode == 0 and env.get("RAW_FILE"):
        return env, "fast"
    if result.returncode == 0 and "SKIPPED_DUPLICATE" in result.stdout:
        return env, "skipped"

    print(f"Fast mode failed, falling back to {engine}...")
    script = "transcribe_whisperx.py" if engine == "whisperx" else "transcribe_whisper.py"
    result = run_script(script, [url])
    env = parse_env(result.stdout)
    if result.returncode != 0 or not env.get("RAW_FILE"):
        print(result.stderr.strip() or result.stdout.strip() or "Error: transcription failed")
        sys.exit(1)
    return env, engine


def load_metadata(channel_dir: str, video_id: str) -> dict:
    path = os.path.join(channel_dir, "raw", f"metadata_{video_id}.json")
    try:
        import json

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def build_summary(env: dict, video_id: str, depth: str) -> str:
    from mistral_client import MistralClient
    from summarize import summarize_transcript

    transcript_file = env.get("ENRICHED_FILE") or env["RAW_FILE"]
    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read()
    client = MistralClient()
    summary = summarize_transcript(
        client,
        transcript,
        env.get("TITLE", "Unknown title"),
        load_metadata(env["CHANNEL_DIR"], video_id).get("channel", "Unknown channel"),
        env.get("SOURCE_LANG", "en"),
        depth,
    )
    path = os.path.join(env["CHANNEL_DIR"], f"summary_{video_id}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"Summary saved: {path}")
    return path


def build_graph(env: dict, video_id: str, depth: str) -> tuple[str, int]:
    from extract_triplets import extract_triplets
    from mistral_client import MistralClient

    transcript_file = env.get("ENRICHED_FILE") or env["RAW_FILE"]
    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read()
    channel_dir = env["CHANNEL_DIR"]
    graph_path = os.path.join(channel_dir, "graph.json")

    existing_entities = None
    if os.path.exists(graph_path):
        import json

        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            existing_entities = [n.get("id", "") for n in data.get("nodes", [])]
        except (OSError, ValueError):
            existing_entities = None

    client = MistralClient()
    triplets = extract_triplets(
        client,
        transcript,
        load_metadata(channel_dir, video_id).get("channel", "Unknown channel"),
        depth,
        existing_entities,
    )
    triplets_path = os.path.join(channel_dir, f"triplets_{video_id}.json")
    import json

    with open(triplets_path, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
    print(f"Triplets saved: {triplets_path} ({len(triplets)} extracted)")

    result = run_script("graph_extractor.py", [channel_dir, triplets_path])
    if result.returncode != 0:
        print(result.stderr.strip() or "Error: graph generation failed")
        sys.exit(1)
    html_match = re.search(r"Visualization:\s*(\S+)", result.stdout)
    return html_match.group(1) if html_match else os.path.join(channel_dir, "graph.html"), len(triplets)


def export_obsidian(env: dict, video_id: str) -> int:
    channel_dir = env["CHANNEL_DIR"]
    triplets_path = os.path.join(channel_dir, f"triplets_{video_id}.json")
    metadata_path = os.path.join(channel_dir, "raw", f"metadata_{video_id}.json")
    out_dir = os.path.join(channel_dir, "obsidian")
    args = [triplets_path, out_dir, "--metadata", metadata_path]
    result = run_script("obsidian_exporter.py", args)
    if result.returncode != 0:
        print(result.stderr.strip() or "Error: Obsidian export failed")
        sys.exit(1)
    match = re.search(r"Created (\d+) Obsidian notes", result.stdout)
    return int(match.group(1)) if match else 0


def process_video(url: str, depth: str, engine: str, obsidian: bool) -> dict:
    video_id = extract_video_id(url)
    if not video_id:
        print(f"Error: could not extract video ID from {url}")
        return {"status": "error"}

    print(f"\n=== {url} ===")
    if is_duplicate(video_id):
        print("SKIPPED: already processed")
        return {"status": "skipped", "title": "", "url": url}

    env, mode = transcribe(url, engine)
    if mode == "skipped":
        print("SKIPPED: already processed")
        return {"status": "skipped", "title": env.get("TITLE", ""), "url": url}

    summary_path = build_summary(env, video_id, depth)
    graph_html, triplet_count = build_graph(env, video_id, depth)
    notes = export_obsidian(env, video_id) if obsidian else 0
    run_script("generate_video_db.py", [])

    return {
        "status": "processed",
        "title": env.get("TITLE", ""),
        "url": url,
        "summary": summary_path,
        "graph_html": graph_html,
        "triplets": triplet_count,
        "obsidian_notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Turn a YouTube video into transcript, summary, and knowledge graph (Mistral AI)"
    )
    parser.add_argument("urls", nargs="+", help="YouTube video or playlist URL(s)")
    parser.add_argument("--depth", choices=("light", "standard", "deep"), default="standard")
    parser.add_argument("--engine", choices=("whisper", "whisperx"), default="whisper")
    parser.add_argument("--obsidian", action="store_true")
    args = parser.parse_args()

    urls = []
    for url in args.urls:
        if "playlist?list=" in url:
            urls.extend(expand_playlist(url))
        else:
            urls.append(url)

    print(f"Processing {len(urls)} video(s) at depth={args.depth}")
    results = []
    for url in urls:
        results.append(process_video(url, args.depth, args.engine, args.obsidian))

    processed = [r for r in results if r["status"] == "processed"]
    skipped = [r for r in results if r["status"] == "skipped"]
    print(f"\nDone: {len(processed)} processed, {len(skipped)} skipped, "
          f"{len(results) - len(processed) - len(skipped)} error(s)")

    if len(results) == 1 and results[0]["status"] == "processed":
        r = results[0]
        print(f"\nTitle: {r['title']}")
        print(f"Depth: {args.depth}")
        print(f"Summary: {r['summary']}")
        print(f"Graph: {r['graph_html']} ({r['triplets']} triplets)")
        if args.obsidian:
            print(f"Obsidian notes: {r['obsidian_notes']} in obsidian/")
        print(f"Open the graph: open \"{r['graph_html']}\"")
    elif len(results) > 1:
        print("\n| Video | Status |")
        print("| :--- | :--- |")
        for r in results:
            title = r.get("title", r.get("url", "?")).replace("|", "\\|")
            print(f"| {title} | {r['status']} |")
        for r in processed:
            print(f"\nGraph: {r['graph_html']}")


if __name__ == "__main__":
    main()

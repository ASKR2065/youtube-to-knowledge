# youtube-to-knowledge

Turn any YouTube video into a transcript, summary, and knowledge graph — powered by the Mistral AI API.

Paste a link. Get structured knowledge.

## How it works

1. Fetches the transcript (YouTube API or local Whisper)
2. Generates a structured summary in the video's language (Mistral AI)
3. Extracts entities and relationships → builds an interactive knowledge graph (Mistral AI)

## Requirements

- Python 3.10+
- A [Mistral AI API key](https://console.mistral.ai/) (free tier available)
- ffmpeg (required for Whisper mode)

### Install ffmpeg

| OS | Command |
|----|---------|
| macOS | `brew install ffmpeg` |
| Ubuntu / Debian | `sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |
| Windows | `winget install ffmpeg` |

## Setup

```bash
git clone https://github.com/ASKR2065/youtube-to-knowledge
cd youtube-to-knowledge
pip install -r requirements.txt
```

Create your `.env` file with your Mistral API key:

```bash
cp .env.example .env
# then edit .env and paste your key
```

Alternatively, export the key in your shell:

```bash
export MISTRAL_API_KEY="your_key_here"
```

## Usage

Process one video end-to-end (transcript → summary → knowledge graph):

```bash
python src/process.py https://www.youtube.com/watch?v=VIDEO_ID
```

Multiple videos at once:

```bash
python src/process.py https://youtube.com/watch?v=abc https://youtube.com/watch?v=def
```

An entire playlist:

```bash
python src/process.py https://www.youtube.com/playlist?list=PLxxx
```

Already-processed videos are skipped automatically.

### Options

```
python src/process.py [--depth light|standard|deep] [--engine whisperx|whisper] [--obsidian] <URL> [<URL2> ...]
```

| Flag | Values | Default | Effect |
|------|--------|---------|--------|
| `--depth` | `light`, `standard`, `deep` | `standard` | Controls triplet count and summary detail |
| `--engine` | `whisper`, `whisperx` | `whisper` | Selects transcription engine for fallback |
| `--obsidian` | — | off | Exports entity notes to `obsidian/` subfolder |

Examples:

```bash
python src/process.py --depth deep https://www.youtube.com/watch?v=VIDEO_ID
python src/process.py --engine whisperx --depth light https://www.youtube.com/watch?v=VIDEO_ID
python src/process.py --obsidian https://www.youtube.com/watch?v=VIDEO_ID
```

### Ask questions about a processed video

```bash
python src/ask.py <video_id_or_channel> "What did the speaker say about pricing?"
```

### Run the LLM steps individually

Summary only:

```bash
python src/summarize.py vault/content/<channel>/raw/transcript_<video_id>.txt \
  --title "Video title" --channel "Channel" --language en --depth standard \
  --output vault/content/<channel>/summary_<video_id>.md
```

Triplet extraction only:

```bash
python src/extract_triplets.py vault/content/<channel>/raw/transcript_<video_id>.txt \
  --channel "Channel" --depth standard \
  --graph vault/content/<channel>/graph.json \
  --output vault/content/<channel>/triplets_<video_id>.json
python src/graph_extractor.py vault/content/<channel> vault/content/<channel>/triplets_<video_id>.json
```

## Output

Files are saved to `vault/content/<channel_name>/`:

```
raw/
  transcript_<video_id>.txt    raw transcript
  metadata_<video_id>.json     title, channel, video ID, language
summary_<video_id>.md          structured summary
triplets_<video_id>.json       knowledge graph source data (per video)
graph.json                     cumulative graph in node-link format
graph.html                     open in browser — interactive visualization
```

Each file is named after the video ID, so multiple videos from the same channel are stored without overwriting each other.

## Transcription modes

| Mode | Speed | Requirement |
|------|-------|-------------|
| Fast (default) | Seconds | Video must have subtitles |
| Whisper (local) | Minutes | Any video, no subtitles needed |
| WhisperX (opt-in) | Minutes | `pip install -r requirements-whisperx.txt` |

`process.py` tries fast mode first and falls back to Whisper automatically.

### WhisperX mode

WhisperX provides word-level timestamps, speaker diarization, and faster transcription via faster-whisper. Install the extra dependencies:

```bash
pip install -r requirements-whisperx.txt
```

Then use the `--engine` flag:

```bash
python src/process.py --engine whisperx https://www.youtube.com/watch?v=VIDEO_ID
```

Speaker diarization requires a [HuggingFace token](https://huggingface.co/settings/tokens) set as `HF_TOKEN` in your environment. Without it, WhisperX still produces timestamped segments but without speaker labels.

## Obsidian integration

Add `--obsidian` to export the knowledge graph directly into your Obsidian vault.

```bash
python src/process.py --obsidian https://www.youtube.com/watch?v=VIDEO_ID
```

This generates one `.md` file per extracted entity in `vault/content/<channel_name>/obsidian/`. Each file contains:

- **YAML frontmatter** — tags, source title, URL, channel
- **Relations** — outgoing links to other entities: `- made_by: [[Anthropic]]`
- **Referenced by** — incoming links: `- [[Claude]] → made_by`

Wikilinks use Obsidian's `[[filename|display]]` aliasing, so they resolve correctly even when entity names contain special characters.

To connect the exported notes to your main vault, point Obsidian at the `vault/` folder or copy the `obsidian/` directory into your existing vault.

You can also run the exporter directly:

```bash
python src/obsidian_exporter.py vault/content/<channel>/triplets_<video_id>.json /path/to/output \
  --metadata vault/content/<channel>/raw/metadata_<video_id>.json
```

## Model configuration

By default the pipeline uses `mistral-large-latest`. Change it via `.env` or environment:

```bash
export MISTRAL_MODEL="mistral-small-latest"
```

## License

[MIT](LICENSE)

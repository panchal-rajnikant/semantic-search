# Semantic Search

A small Python project that demonstrates semantic search with text embeddings.
It loads text documents, splits them into overlapping word-based chunks, converts
the chunks into vector embeddings, saves the vector index, and returns the
chunks most similar to a user's query.

## How It Works

1. Text files are loaded from the `data/` directory by `ingest.py`.
2. Each document is split into chunks of 100 words with 20 words of overlap.
3. `all-MiniLM-L6-v2` converts each chunk into a numerical embedding vector.
4. The embeddings and chunk metadata are saved in the `storage/` directory.
5. `main.py` loads the saved index and accepts a search query.
6. The query is converted into an embedding using the same model.
7. Cosine similarity compares the query embedding with every chunk embedding.
8. Results are sorted by similarity score and the top three are displayed.

Semantic search compares meaning rather than requiring an exact keyword match.

## Project Structure

```text
semantic-search/
├── data/                 # Input .txt documents
├── ingest.py             # Build or rebuild the vector index
├── src/
│   ├── chunker.py        # Split documents into overlapping chunks
│   ├── embedding_service.py # Generate embeddings with Sentence Transformers
│   ├── ingestion.py      # Load, chunk, embed, and save data
│   ├── loader.py         # Load text documents
│   ├── search_service.py # Search the saved vector index
│   └── vector_store.py   # Save and load embeddings and metadata
├── main.py               # Query application entry point
├── storage/              # Generated vector index files
├── requirements.txt      # Python dependencies
└── README.md
```

## Requirements

- Python 3.10 or newer
- Internet access on the first run so Sentence Transformers can download the
	model

## Installation

Clone the repository and open its directory:

```bash
git clone <repository-url>
cd semantic-search
```
Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run all commands from the repository root.

### Build the Index

Run ingestion before the first search:

```bash
python ingest.py
```

This creates the generated files below:

```text
storage/
├── embeddings.npy
└── metadata.json
```

Run `ingest.py` again whenever files in `data/` are added, removed, or edited.
It rebuilds the saved embeddings and metadata so search results reflect the
current data.

### Search the Index

```bash
python main.py
```

The program prompts for a query. To search different content, update the `.txt`
files in `data/`, run `python ingest.py`, and then run `python main.py` again.

The first run may download the `all-MiniLM-L6-v2` model. The model is loaded
locally after that, depending on your Sentence Transformers cache configuration.

## Example Output

```text
Enter search query: How can I build APIs using Python?
Rank: 1
Score: 0.8123
Source: fastapi.txt
```

The exact number of chunks and similarity scores depends on the input files.

## Generated Files

The `storage/` directory contains generated index files and should not be edited
manually. Rebuild it with `python ingest.py` whenever the source data changes.
These files are ignored by Git.

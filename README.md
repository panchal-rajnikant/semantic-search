# Semantic Search

A small Python project that demonstrates semantic search with text embeddings.
It loads text documents, splits them into overlapping word-based chunks, converts
the chunks into vector embeddings, and returns the chunks most similar to a
user's query.

## How It Works

1. Text files are loaded from the `data/` directory.
2. Each document is split into chunks of 50 words with 10 words of overlap.
3. `all-MiniLM-L6-v2` converts each chunk into a numerical embedding vector.
4. The query is converted into an embedding using the same model.
5. Cosine similarity compares the query embedding with every chunk embedding.
6. Results are sorted by similarity score and the top three are displayed.

Semantic search compares meaning rather than requiring an exact keyword match.

## Project Structure

```text
semantic-search/
├── data/                 # Input .txt documents
├── src/
│   ├── chunker.py        # Split documents into overlapping chunks
│   ├── embeddings.py     # Generate embeddings with Sentence Transformers
│   ├── loader.py         # Load text documents
│   └── search.py         # Rank chunks with cosine similarity
├── main.py               # Application entry point
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

Run the application from the repository root:

```bash
python main.py
```

To search different content, add `.txt` files to `data/` and change the query
in `main.py`.

## Example Output

```text
Loaded 3 documents
Created 12 chunks
Embedding shape: (12, 384)

Query: How can I build APIs using Python?
Rank: 1
Score: 0.8123
Source: fastapi.txt
```

The exact number of chunks and similarity scores depends on the input files.

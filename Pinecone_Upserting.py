
import os

PINECONE_API_KEY    = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_HOST = os.environ["PINECONE_INDEX_HOST"]
PDF_FOLDER          = os.environ.get("PDF_FOLDER", "./Source-Docx")
NAMESPACE           = "avivo-assessment-suryaa"

import fitz                          # PyMuPDF
import pandas as pd
import tiktoken
import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
def extract_text_from_pdfs(folder_path):
    """Read every PDF in folder_path, return a DataFrame with FileName + TextContent."""
    data_list = []

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {folder_path}")

    for filename in pdf_files:
        pdf_path = os.path.join(folder_path, filename)
        full_text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    full_text += page.get_text() + "\n\n"
            data_list.append({"FileName": filename, "TextContent": full_text.strip()})
            print(f" {filename}")
        except Exception as e:
            print(f"{filename} — ERROR: {e}")
            data_list.append({"FileName": filename, "TextContent": f"ERROR: {e}"})

    df = pd.DataFrame(data_list)
    return df
def build_chunks(df, max_tokens=256, min_tokens=175, overlap=20):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def chunk_text(text):
        tokens = encoding.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end   = start + max_tokens
            chunk = tokens[start:end]
            if len(chunk) < min_tokens:
                break                        
            chunks.append(encoding.decode(chunk))
            start += max_tokens - overlap
        return chunks

    rows = []
    for _, row in df.iterrows():
        if row["TextContent"].startswith("ERROR:"):
            continue
        chunks = chunk_text(row["TextContent"])
        for i, chunk in enumerate(chunks):
            rows.append({
                "FileName":   row["FileName"],
                "ChunkIndex": i,
                "ChunkText":  chunk,
                "VectorID":   f"{os.path.splitext(row['FileName'])[0]}_chunk_{i}"
            })

    df_chunks = pd.DataFrame(rows)
    print(f"   Total chunks produced: {len(df_chunks)}")
    return df_chunks
def embed_chunks(df_chunks):
    tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
    model     = AutoModel.from_pretrained("BAAI/llm-embedder")
    model.eval()

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    embeddings = []
    for i, row in df_chunks.iterrows():
        embeddings.append(get_embedding(row["ChunkText"]))
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(df_chunks)} done")

    df_chunks = df_chunks.copy()
    df_chunks["Embedding"] = embeddings
    print("   All chunks embedded.")
    return df_chunks
def upsert_to_pinecone(df_chunks, namespace):
    """Build Pinecone vectors and upsert in batches of 100."""
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_INDEX_HOST)

    vectors = []
    for _, row in df_chunks.iterrows():
        vectors.append({
            "id":     row["VectorID"],
            "values": row["Embedding"],
            "metadata": {
                "file_name":   row["FileName"],
                "chunk_index": int(row["ChunkIndex"]),
                "chunk_text":  row["ChunkText"]
            }
        })
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(namespace=namespace, vectors=batch)
        print(f"   Upserted batch {i // batch_size + 1}  ({len(batch)} vectors)")

    print(f"\n Done! {len(vectors)} vectors upserted into namespace '{namespace}'.")

if __name__ == "__main__":
    df        = extract_text_from_pdfs(PDF_FOLDER)
    df_chunks = build_chunks(df)
    df_chunks = embed_chunks(df_chunks)
    upsert_to_pinecone(df_chunks, NAMESPACE)

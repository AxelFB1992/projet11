"""
Construit (ou reconstruit) la base vectorielle FAISS à partir des événements
nettoyés. Ce script peut être relancé à tout moment pour régénérer l'index
depuis data/events_clean.csv.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_PATH = "data/events_clean.json"
INDEX_PATH = "vectorstore/faiss_index"


def load_documents(path=DATA_PATH):
        """Charge les événements nettoyés et les convertit en Documents LangChain,
        avec les métadonnées séparées du texte vectorisé (voir discussion sur
        l'exclusion de la date de l'embedding_text)."""
        df = pd.read_json(path)
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce", utc=True)
        df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce", utc=True)
    
        documents = []
        for _, row in df.iterrows():
            metadata = {
                "uid": str(row["uid"]),
                "title": row["title"],
                "city": row["city"],
                "department": row["department"],
                "region": row["region"],
                "date_start": row["date_start"].isoformat() if pd.notna(row["date_start"]) else None,
                "date_end": row["date_end"].isoformat() if pd.notna(row["date_end"]) else None,
                "venue_name": row["venue_name"],
                "latitude": row["latitude"] if pd.notna(row["latitude"]) else None,
                "longitude": row["longitude"] if pd.notna(row["longitude"]) else None,
                "url": row["url"],
            }
            documents.append(Document(page_content=row["embedding_text"], metadata=metadata))
    
        return documents


def chunk_documents(documents):
    """Découpe les documents en chunks. Pour ce corpus (textes courts par
    événement), la grande majorité des documents tiendront dans un seul
    chunk — le splitter protège simplement les rares descriptions plus
    longues que la moyenne, sans intervention manuelle au cas par cas."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " | ", ". ", " "],
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    """Vectorise les chunks avec Mistral Embed et construit l'index FAISS.

    Note sur le type d'index : FAISS via LangChain utilise par défaut un
    IndexFlatL2 (recherche exacte par force brute). Pour ~120 vecteurs,
    c'est optimal — les index approximatifs (IVF, HNSW) n'apportent un
    gain qu'à partir de dizaines de milliers de vecteurs, au prix d'une
    perte de précision. Point à documenter dans le rapport technique
    comme recommandation pour un passage à l'échelle en production."""
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def test_search(vectorstore):
    """Quelques recherches de vérification pour valider la pertinence sémantique."""
    test_queries = [
        "Un concert de musique classique à Nantes",
        "Exposition sur le patrimoine régional",
        "Spectacle pour enfants en Vendée",
        "Événement autour de la photographie",
    ]
    print("\n" + "=" * 70)
    print("TESTS DE RECHERCHE")
    print("=" * 70)
    for query in test_queries:
        print(f"\nRequête : « {query} »")
        results = vectorstore.similarity_search_with_score(query, k=3)
        for doc, score in results:
            print(f"  [score={score:.4f}] {doc.metadata['title']} — {doc.metadata['city']} ({doc.metadata['department']})")


def main():
    print("Chargement des événements...")
    documents = load_documents()
    print(f"  {len(documents)} événements chargés")

    print("Découpage en chunks...")
    chunks = chunk_documents(documents)
    print(f"  {len(chunks)} chunks générés")

    print("Vectorisation avec Mistral Embed et construction de l'index FAISS...")
    vectorstore = build_vectorstore(chunks)
    print(f"  {vectorstore.index.ntotal} vecteurs indexés")

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    vectorstore.save_local(INDEX_PATH)
    print(f"  Index sauvegardé dans {INDEX_PATH}")

    test_search(vectorstore)


if __name__ == "__main__":
    main()

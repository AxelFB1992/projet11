"""
Script d'exploration manuelle — vérifie visuellement le comportement du
filtrage par métadonnées combiné à la recherche sémantique FAISS.
Ce n'est PAS un test automatisé (pas d'assertions) : les résultats sont
à examiner à l'œil pour juger de la pertinence. Nécessite un index FAISS
déjà construit (vectorize_events.py) et une clé API Mistral valide.
"""

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

INDEX_PATH = "vectorstore/faiss_index"


def load_vectorstore():
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


def main():
    vectorstore = load_vectorstore()
    total_docs = len(vectorstore.index_to_docstore_id)
    print(f"Corpus total indexé : {total_docs} vecteurs\n")

    print("=" * 70)
    print("SANS filtre — « Spectacle pour enfants »")
    print("=" * 70)
    for doc, score in vectorstore.similarity_search_with_score("Spectacle pour enfants", k=5):
        print(f"  [score={score:.4f}] {doc.metadata['title']} — {doc.metadata['department']}")

    print("\n" + "=" * 70)
    print("AVEC filtre department=Vendée — « Spectacle pour enfants »")
    print("=" * 70)
    results = vectorstore.similarity_search_with_score(
        "Spectacle pour enfants", k=5, filter={"department": "Vendée"}, fetch_k=total_docs
    )
    if not results:
        print("  Aucun résultat.")
    for doc, score in results:
        print(f"  [score={score:.4f}] {doc.metadata['title']} — {doc.metadata['department']}")

    print("\n" + "=" * 70)
    print("AVEC filtre department=Loire-Atlantique — « Exposition photographie »")
    print("=" * 70)
    results = vectorstore.similarity_search_with_score(
        "Exposition photographie", k=5, filter={"department": "Loire-Atlantique"}, fetch_k=total_docs
    )
    for doc, score in results:
        print(f"  [score={score:.4f}] {doc.metadata['title']} — {doc.metadata['department']}")


if __name__ == "__main__":
    main()

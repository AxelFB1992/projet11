""".
Construit (ou reconstruit) la base vectorielle FAISS à partir des événements
nettoyés. Ce script peut être relancé à tout moment pour régénérer l'index
depuis data/events_clean.json.

Dans cet index, il y a deux technologie :
-les métadonnées comme la date
-La recherche semantique qui utilise l'index vectorisé pour faire des recherches sur les champs textuels

Teste également le filtrage par métadonnées combiné à la recherche sémantique FAISS
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

#Permet de charger les variables d'environnement
load_dotenv()

#Le fichier où l'on récupère les éléments
DATA_PATH = "data/events_clean.json"

#Le fichier où l'on va stocker l'index
INDEX_PATH = "vectorstore/faiss_index"

def load_documents(path=DATA_PATH):
    """Charge les événements nettoyés et les convertit en Documents LangChain,
    avec les métadonnées séparées du texte vectorisé.
    Autrement dit, on génère des documents qui vont avoir la structure suivante :
        -le contenu de embedding_text en format brut : une ligne
        -les metadonnées sous forme de sous document structuré avec plusieurs champs reprenant les champs du fichier (hors embedding)
    """
    
    #On lit le fichier json correspondant aux évenements nettoyés et on les stocke dans un dataframe (ils sont déjà parsé).
    df = pd.read_json(path)
    #On récupère tout de suite les dates de début et les dates de fin qu'on l'on convertit en datetime
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

    #On liste cette liste de couple de documents
    return documents


def chunk_documents(documents):
    """Découpe les documents en chunks. Pour ce corpus (textes courts par
    événement), la grande majorité des documents tiendront dans un seul
    chunk — le splitter protège simplement les rares descriptions plus
    longues que la moyenne, sans intervention manuelle au cas par cas."""

    """Par ailleurs cette fonction découpe uniquement la partie embedding_text et génère des chunks par rapport à cette partie
    Le metadata original est copié tel quel sur chaque chunk généré correspondant à sa partie embedding_text
    """
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

    #retourne un moteur "d'embedding" permettant de faire la vectorisation via l'API Mistral
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    
    #Utilisation d'un objet de classe FAISS et de la méthode from document pour lancer les différents appels et faire la vectorisation
    #Cette méthode retourne un objet FAISS (vectorstore) qui contient : l'index vectorisé, le document original et le docstore
    #l'index FAISS contient des vecteurs qui permettent de faire des recherches dans les documents
    #Une reférence vers embeddings (le moteur permettant de faire la vectorisation)
    #le docstore qui fait le lien entre l'index et le document original.
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

#Permet de faire des tests de recherche pour voir si des correspondances sont bien trouvés entre les questions et l'index
#On re-utilise l'objet vectorstore car c'est lui qui a permet de vectorisé l'index, donc qui vectorisera aussi les questions
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

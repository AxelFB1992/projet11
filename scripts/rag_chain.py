"""
Chaîne RAG complète : recherche vectorielle FAISS + génération Mistral.
Intègre une extraction automatique du lieu mentionné dans la question,
pour filtrer la recherche par métadonnées avant la génération.
"""

import os
import time
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

INDEX_PATH = "vectorstore/faiss_index"

SYSTEM_PROMPT = """Tu es l'assistant virtuel de Puls-Events, une plateforme de découverte d'événements culturels en Pays de la Loire.

Règles strictes :
- Réponds UNIQUEMENT à partir des événements fournis dans le contexte ci-dessous.
- Si aucun événement du contexte ne correspond à la question, dis-le clairement : "Je n'ai pas trouvé d'événement correspondant à votre demande dans notre base actuelle." N'invente jamais d'événement, de date ou de lieu.
- Si l'utilisateur demande un lieu précis (ville ou département) et qu'aucun événement du contexte ne s'y trouve réellement, ne présente JAMAIS un événement d'un autre lieu comme s'il y correspondait. Dis explicitement qu'aucun événement n'est disponible à cet endroit précis, en mentionnant éventuellement où les événements similaires ont lieu.
- Sois concis et chaleureux, comme un conseiller culturel local.
- Présente toujours les événements sous forme de liste numérotée (1., 2., 3., ...), jamais avec des tirets ou puces.
- Pour chaque événement recommandé, mentionne son titre, sa date, sa ville, et le lien si disponible.
- Réponds toujours en français.

Contexte (événements disponibles) :
{context}
"""

#De la même manière qu'on rechargait le vectorstore pour faire des tests avec des questions dans vectorize_events.py
#On recharge ici cet objet FAISS pour avoir accès au même moteur de vectorisation que celui a permis de faire l'index FAISSle Document original contient toutes les informations de départ
def load_vectorstore():
    #On crée le type d'objet qui va recevoir le vectorstore
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    #Et on va charger le vrai modèle que l'on a stocké dans INDEX_PATH pour retourner un objet FAISS complet
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

#le Document original contient toutes les informations de départ
#On recupère, via l'index, les différents départements et villes qui sont présents dans les documents originaux
def load_known_locations(vectorstore):
    """Construit les ensembles de villes et départements réellement présents
    dans le corpus indexé, pour la détection de lieu dans les questions."""
    departments, cities = set(), set()
    for doc_id in vectorstore.index_to_docstore_id.values():
        doc = vectorstore.docstore.search(doc_id)
        if doc.metadata.get("department"):
            departments.add(doc.metadata["department"])
        if doc.metadata.get("city"):
            cities.add(doc.metadata["city"])
    return departments, cities


def extract_location(question, departments, cities):
    """Détecte un département ou une ville mentionné dans la question
    (correspondance simple insensible à la casse). Le département est
    vérifié en premier — plus fiable qu'une ville homonyme d'un autre lieu."""
    question_lower = question.lower()
    for dept in departments:
        if dept.lower() in question_lower:
            return {"department": dept}
    for city in cities:
        if city.lower() in question_lower:
            return {"city": city}
    return None


def format_docs(docs):
    if not docs:
        return "Aucun événement trouvé."
    blocs = []
    for doc in docs:
        meta = doc.metadata
        bloc = (
            f"- {meta.get('title')}\n"
            f"  Lieu : {meta.get('venue_name', '')}, {meta.get('city')} ({meta.get('department')})\n"
            f"  Date : {meta.get('date_start')}\n"
            f"  Lien : {meta.get('url', 'non disponible')}\n"
            f"  Description : {doc.page_content}"
        )
        blocs.append(bloc)
    return "\n\n".join(blocs)


def build_rag_chain(k=5):
    #On recupère l'objet FAISS encapsulant l'index et les documents originaux
    vectorstore = load_vectorstore()
    #On récupère, via l'index, les différents départements et villes contenus dans les documents
    departments, cities = load_known_locations(vectorstore)
    #On liste le nombre total de documents
    total_docs = len(vectorstore.index_to_docstore_id)

    #Cette méthode permet de retourner les similarités entre la question et les documents présent dans l'objet (via l'index)
    #Avant de rechercher directement les similarités, on va d'abord regarder les départements et villes énoncées dans la question
    #correspondent bien à des villes et départements présents dans le corpus (avec avec load_knows_locations)
    #Si c'est bien le cas, alors on va rechercher des similarités avec un élement en plus : un filtre pour la localisation
    #Si ce n'est pas le cas, alors on fait juste une recherche de similarité vu qu'aucune localisation connue n'est mentionnée
    def retrieve(question):
        location_filter = extract_location(question, departments, cities)
        if location_filter:
            results = vectorstore.similarity_search(
                question, k=k, filter=location_filter, fetch_k=total_docs
            )
            #Si il n'y a pas de resultat pour la recherche filtré (trop restrictives), alors on donne quand même un résultat
            #Mais en signalant, via le LLM, que les lieux sont proches mais non correspondant à ceux demandés (via le prompt système) 
            if not results:
                # Aucun événement dans ce lieu précis : on redonne quand même
                # un contexte général (sans filtre) pour que le modèle puisse
                # au moins signaler où se trouvent les événements proches —
                # le prompt système lui interdit de les présenter comme
                # correspondant au lieu demandé.
                results = vectorstore.similarity_search(question, k=k)
        else:
            #Aucune localisation connue mentionné, on fait un simple recherche par similarité
            results = vectorstore.similarity_search(question, k=k)
        
        print(f"  [DEBUG] {len(results)} documents récupérés : {[d.metadata['title'] for d in results]}")

        #On retourne les résultats d'évenements formattés proprement grâce à la méthode format_docs, pour une meilleur visibilité
        return format_docs(results)

    #====C'est cette partie qui correspond veritablement à l'interaction entre les requêtes et le modèle====

    #Ici c'est le prompt composé du prompt système et de la question
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    #Ici c'est le LLM fournit par Mistral, un modèle de langage déjà entrainé sur une quantité de données suffisante pour répondre
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0.3)

    #Voilà le système RAG complet condensés dans ces quelques lignes qui contient :
    # - le promt global (question + prompt système)
    # - le llm
    # - le contexte, qui consiste en un appel à la méthode retrieve qui se charge de faire appel à la base de données vectorielle
    #ainsi que tous les documents originaux. De même la question de l'utilisation fait également partie du contexte.
    # - Un parseur, qui permet certainement de structurer la réponse fournie par le llm
    chain = (
        {"context": RunnableLambda(retrieve), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    #On retourne ce système qui servira à répondre aux différentes questions posées
    #En sachant qu'à chaque question, ça sera un appel interne à la fonction invoke(question) qui permettra d'utiliser toutes les 
    #ressources définies à l'intérieur de celui-ci (contexte, prompt, llm)
    return chain


def main():
    chain = build_rag_chain()

    test_questions = [
        "Quels concerts de musique classique se jouent à Nantes ?",
        "Y a-t-il des expositions sur le patrimoine en ce moment ?",
        "Quels spectacles pour enfants sont prévus en Vendée ?",
        "Y a-t-il des événements sportifs prévus ?",
    ]

    for question in test_questions:
        print("=" * 70)
        print(f"Q : {question}")
        print("-" * 70)
        response = chain.invoke(question)
        print(f"R : {response}\n")


if __name__ == "__main__":
    main()

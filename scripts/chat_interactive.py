"""
Chat interactif en ligne de commande pour tester le chatbot RAG Puls-Events.
Tape ta question et appuie sur Entrée. Tape 'quit' ou 'exit' pour arrêter.
"""

from rag_chain import build_rag_chain


def main():
    """C'est exactement le même processus que dans l'autre script, cependant on remet en boucle un input pour une entrée utilisateur
    Sauf quand celui-ci écrit explicitement "quit", "exit" ou "q"
    Mais quand il ecrit autre chose, on invoque tout simplement la fonction 'invoke' de l'objet chain qui contient notre systeme RAG
    Celle-ci va utiliser notre objet pour obtenir une réponse grâce au contexte, à l'input et au LLM
    """
    
    print("=" * 70)
    print("Chatbot Puls-Events — POC RAG (Pays de la Loire)")
    print("Tape 'quit' ou 'exit' pour arrêter.")
    print("=" * 70)

    print("\nChargement de la base vectorielle et du modèle...")
    chain = build_rag_chain()
    print("Prêt !\n")

    while True:
        try:
            question = input("Vous : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nÀ bientôt !")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("À bientôt !")
            break

        try:
            response = chain.invoke(question)
            print(f"\nBot : {response}\n")
        except Exception as e:
            print(f"\n[Erreur] {e}\n")


if __name__ == "__main__":
    main()

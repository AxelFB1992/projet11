# test_setup.py
"""Vérifie que l'environnement RAG est correctement configuré."""

import sys
from importlib.metadata import version

def test_imports():
    try:
        import langchain
        print(f"✓ langchain {version('langchain')}")

        import langchain_community
        print(f"✓ langchain_community {version('langchain-community')}")

        import langchain_mistralai
        print(f"✓ langchain_mistralai {version('langchain-mistralai')}")

        import faiss
        print(f"✓ faiss {version('faiss-cpu')}")

        import mistralai
        print(f"✓ mistralai {version('mistralai')}")

        from langchain_community.vectorstores import FAISS
        from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
        print("✓ Imports clés (FAISS, ChatMistralAI, MistralAIEmbeddings) OK")

        return True
    except ImportError as e:
        print(f"✗ Erreur d'import : {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

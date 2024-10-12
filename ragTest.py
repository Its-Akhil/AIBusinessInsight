import os
from rag_knowledge_base import RAGKnowledgeBase


def main():
    kb = RAGKnowledgeBase()

    query = "What is the agency accelerator about?"
    print(f"Query: {query}")

    results, indices = kb.query_knowledge_base(query, k=5)

    if results:
        print("\nTop results:")
        for i, result in enumerate(results[:5], 1):
            print(f"\nResult {i}:")
            print(f"- Content: {result['text'][:200]}...")
            print(f"- Document: {result['document']}")
            print(f"- Semantic Distance: {result['semantic_distance']}")
            print(f"- BM25 Score: {result['bm25_score']}")
    else:
        print("No results found.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

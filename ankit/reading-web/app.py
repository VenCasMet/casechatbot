from chatbot import (
    debug_case,
    load_documents,
    find_relevant_documents,
    ask_llm
)

documents = load_documents()
debug_case("OA/673/2024", documents)
print(f"\nLoaded {len(documents)} documents.\n")

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    relevant_docs = find_relevant_documents(
        question,
        documents
    )

    if not relevant_docs:

        print("\nNo relevant document found.\n")

        continue

    print("\nUsing:\n")

    for doc in relevant_docs:

        print(f"{doc['filename']}   Score : {doc['score']}")

    print()

    answer = ask_llm(
        question,
        relevant_docs
    )

    print("\nAI:\n")

    print(answer)
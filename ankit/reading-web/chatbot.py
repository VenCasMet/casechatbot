import ollama
from pathlib import Path
import re
from collections import Counter

# -----------------------------
# Load API Key
# -----------------------------


DOWNLOAD_FOLDER = Path("downloads2")


def load_documents():
    """
    Reads every TXT file and stores it in memory.
    """

    documents = []

    txt_files = DOWNLOAD_FOLDER.glob("*.txt")

    for txt_file in txt_files:

        try:

            content = txt_file.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            documents.append(
                {
                    "filename": txt_file.name,
                    "text": content
                }
            )

        except Exception as e:

            print(f"Could not read {txt_file.name}: {e}")

    return documents

def debug_case(case_number, documents):

    for doc in documents:

        if case_number in doc["text"]:

            print("\nFOUND EXACTLY IN:", doc["filename"])
            return

    print("\nExact text not found.")


STOP_WORDS = {
    "the", "is", "are", "who", "what", "when", "where",
    "why", "how", "show", "list", "tell", "give",
    "find", "today", "all", "me", "for", "of",
    "to", "in", "on", "with", "a", "an", "and"
}

# -----------------------------
# Regex Patterns
# -----------------------------

CASE_PATTERNS = [

    r"EX-A/\d+/\d+",

    r"OA/\d+/\d+",

    r"TA/\d+/\d+",

    r"MA/\d+/\d+",

    r"DY-NO/\d+/\d+",

    r"DY/\d+/\d+",

]

DATE_PATTERNS = [

    r"\d{2}[.-]\d{2}[.-]\d{4}",

    r"\d{2}/\d{2}/\d{4}"

]

def extract_case_numbers(question):

    cases = []

    for pattern in CASE_PATTERNS:

        found = re.findall(
            pattern,
            question,
            flags=re.IGNORECASE
        )

        cases.extend(found)

    return cases

def extract_dates(question):

    dates = []

    for pattern in DATE_PATTERNS:

        found = re.findall(
            pattern,
            question
        )

        dates.extend(found)

    return dates

def extract_names(question):

    words = re.findall(
        r"[A-Za-z]+",
        question
    )

    names = []

    for word in words:

        if len(word) > 2:

            names.append(word.lower())

    return names

def contains_whole_word(word, text):

    return re.search(

        rf"\b{re.escape(word)}\b",

        text,

        flags=re.IGNORECASE

    ) is not None



def extract_keywords(question):

    words = re.findall(r"[A-Za-z0-9]+", question.lower())

    keywords = []

    for word in words:

        if word not in STOP_WORDS and len(word) > 2:

            keywords.append(word)

    return keywords

def score_document(doc, question):

    text = doc["text"]

    score = 0

    # -----------------
    # Case Number
    # -----------------

    case_numbers = extract_case_numbers(question)

    for case in case_numbers:

        if case.lower() in text.lower():

            score += 1000

    # -----------------
    # Dates
    # -----------------

    dates = extract_dates(question)

    for date in dates:

        if date in text:

            score += 400

    # -----------------
    # Army Air Force Navy
    # -----------------

    lower_question = question.lower()

    if "army" in lower_question:

        score += text.lower().count("army") * 40

    if "air force" in lower_question:

        score += text.lower().count("air force") * 40

    if "navy" in lower_question:

        score += text.lower().count("navy") * 40

    # -----------------
    # Person Names
    # -----------------

    names = extract_names(question)

    for name in names:

        if contains_whole_word(name, text):

            score += 80

    # -----------------
    # Keywords
    # -----------------

    keywords = extract_keywords(question)

    for keyword in keywords:

        matches = re.findall(

            rf"\b{re.escape(keyword)}\b",

            text,

            flags=re.IGNORECASE

        )

        score += len(matches) * 15

    return score

def find_exact_case_documents(case_number, documents):

    matched = []

    parts = case_number.split("/")

    pattern = r"\s*/\s*".join(map(re.escape, parts))

    print("\nSearching for:", case_number)
    print("Regex Pattern:", pattern)

    for doc in documents:

        if re.search(pattern, doc["text"], flags=re.IGNORECASE):

            print("FOUND IN:", doc["filename"])

            matched.append({
                "filename": doc["filename"],
                "text": doc["text"],
                "score": 100000
            })

    print("Matched:", len(matched))

    return matched

def find_relevant_documents(question, documents):

    case_numbers = extract_case_numbers(question)

    if case_numbers:

        exact = find_exact_case_documents(case_numbers[0], documents)

        if exact:
            return exact

    scored_docs = []

    for doc in documents:

        score = score_document(doc, question)

        if score > 0:

            scored_docs.append({

                "filename": doc["filename"],

                "text": doc["text"],

                "score": score

            })

    scored_docs.sort(

        key=lambda x: x["score"],

        reverse=True

    )

    # --------------------------
    # Dynamic Retrieval
    # --------------------------

    case_numbers = extract_case_numbers(question)

    dates = extract_dates(question)

    lower_question = question.lower()

    if case_numbers:

        return scored_docs[:2]

    if dates:

        return scored_docs[:3]

    if "army" in lower_question:

        return scored_docs[:8]

    if "air force" in lower_question:

        return scored_docs[:8]

    if "navy" in lower_question:

        return scored_docs[:8]

    if "list" in lower_question:

        return scored_docs[:8]

    return scored_docs[:5]

def ask_llm(question, relevant_docs):

    if not relevant_docs:
        return "I could not find the answer in the provided documents."

    context = ""

    source_files = []

    for doc in relevant_docs:

        source_files.append(doc["filename"])

        context += f"\n\n========== FILE : {doc['filename']} ==========\n\n"

        context += doc["text"]

    prompt = f"""
You are an AI Assistant for Armed Forces Tribunal (AFT) Cause Lists.

You must answer ONLY using the supplied documents.

==================== RULES ====================

1. Never guess or hallucinate.

2. If the answer is not present in the supplied documents, reply exactly:

I could not find the answer in the provided documents.

3. If the user asks about a specific case number (Example: OA/673/2024 or EX-A/95/2023), search ONLY for that exact case.

4. Never confuse similar case numbers.

5. If multiple documents mention the same case or person, combine the information.

6. If the user asks "Explain this case", explain ONLY what is written in the documents.

7. If the documents do not mention the actual dispute or facts of the case, clearly say:

"The cause list does not contain details of the dispute. It only lists the case information."

8. If the user asks for the next hearing date, answer ONLY if it is explicitly written.

9. Always mention every source file used.

10. Format answers neatly using bullet points whenever possible.

Whenever applicable include:

• Case Number
• Applicant Name
• Respondent
• Court
• Stage
• Case Wing
• Applicant Counsel
• Respondent Counsel
• Source File(s)

==================== DOCUMENTS ====================

{context}

===================================================

Question:

{question}
"""

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": """
You are an expert assistant for Armed Forces Tribunal Cause Lists.

Answer ONLY from the supplied documents.

Never guess.

Never hallucinate.

If information is missing, explicitly say it is unavailable.

Always cite the source files.

If a case number exists, prioritize exact matching over similar cases.

Do not invent hearing dates or case facts.

Return well-structured answers using bullet points.
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0,
            "top_p": 0.9,
            "num_ctx": 16384,
            "num_predict": 1024
        }
    )

    return response["message"]["content"]
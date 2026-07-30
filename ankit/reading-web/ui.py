import streamlit as st

from scraper import run_scraper
from chatbot import (
    load_documents,
    find_relevant_documents,
    ask_llm,
)

# ------------------------------------
# PAGE CONFIG
# ------------------------------------

st.set_page_config(
    page_title="AFT AI Assistant",
    page_icon="⚖️",
    layout="wide",
)

# ------------------------------------
# SESSION STATE
# ------------------------------------

if "documents" not in st.session_state:
    st.session_state.documents = load_documents()

if "messages" not in st.session_state:
    st.session_state.messages = []

documents = st.session_state.documents

# ------------------------------------
# SIDEBAR
# ------------------------------------

with st.sidebar:

    st.title("⚖️ AFT AI Assistant")

    st.success(f"📄 Documents Loaded : {len(documents)}")

    st.divider()

    if st.button("🔄 Update Database", use_container_width=True):

        progress = st.progress(0)

        status = st.empty()

        try:

            status.info("Opening AFT website...")
            progress.progress(10)

            run_scraper()

            progress.progress(80)

            status.info("Reloading Knowledge Base...")

            st.session_state.documents = load_documents()

            documents = st.session_state.documents

            progress.progress(100)

            status.success("✅ Database Updated Successfully!")

        except Exception as e:

            status.error(f"Error : {e}")

    st.divider()

    st.subheader("💡 Example Questions")

    st.markdown("""
- Explain EX-A/95/2023

- List Air Force cases

- List Army cases

- Next hearing of EX-A/95/2023

""")

# ------------------------------------
# TITLE
# ------------------------------------

st.title("⚖️ Armed Forces Tribunal AI")

st.caption("Ask questions from downloaded Cause Lists.")

# ------------------------------------
# DISPLAY CHAT
# ------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------------------------
# CHAT INPUT
# ------------------------------------

question = st.chat_input("Ask your question...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching Documents..."):

            relevant_docs = find_relevant_documents(
                question,
                documents
            )

            if len(relevant_docs) == 0:

                answer = "I could not find any relevant document."

            else:

                with st.expander("📂 Documents Used"):

                    for doc in relevant_docs:

                        st.write(
                            f"**{doc['filename']}** (Score : {doc['score']})"
                        )

                answer = ask_llm(
                    question,
                    relevant_docs
                )

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
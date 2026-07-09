import re
import uuid

import requests
import streamlit as st



st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="⚽️",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000"


# Session state

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_documents" not in st.session_state:
    st.session_state.indexed_documents = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None



# Simple evaluation heuristic

def normalize_words(text: str) -> set[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return set(words)


def calculate_overlap(expected: str, generated: str) -> float:
    expected_words = normalize_words(expected)
    generated_words = normalize_words(generated)

    if not expected_words:
        return 0.0

    return len(expected_words & generated_words) / len(expected_words)



# Header

st.title("RAG Chatboy using Langchain")

st.caption(
    "Upload PDF documents, ask grounded questions, inspect retrieved "
    "evidence, and optionally evaluate answers against expected answers."
)


# Backend health

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=3,
    )

    backend_online = health_response.ok

except requests.RequestException:
    backend_online = False


if backend_online:
    st.success("FastAPI backend connected")
else:
    st.error(
        "FastAPI backend is offline. Start it with: "
        "uvicorn app.main:app --reload"
    )


st.divider()



# Upload section

st.subheader("1. Build Knowledge Base")

uploaded_files = st.file_uploader(
    "Upload one or multiple PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)


col_upload, col_status = st.columns([1, 2])


with col_upload:
    index_button = st.button(
        "Index Documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files or not backend_online,
    )


if index_button:
    multipart_files = [
        (
            "files",
            (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            ),
        )
        for uploaded_file in uploaded_files
    ]

    try:
        with st.spinner(
            "Loading, chunking, embedding, and indexing documents..."
        ):
            response = requests.post(
                f"{API_URL}/ingest",
                files=multipart_files,
                timeout=600,
            )

        if response.ok:
            result = response.json()

            st.session_state.indexed_documents = result.get(
                "documents",
                [],
            )

            st.success(
                f'{result["files_processed"]} document(s) indexed successfully.'
            )

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Files Processed",
                result["files_processed"],
            )

            metric2.metric(
                "New Chunks",
                result["total_new_chunks"],
            )

            metric3.metric(
                "Total Chunks",
                result["total_chunks_in_store"],
            )

        else:
            st.error(
                f"Backend error {response.status_code}: "
                f"{response.text}"
            )

    except requests.RequestException as exc:
        st.error(f"Could not reach backend: {exc}")


if st.session_state.indexed_documents:
    with st.expander(
        "View indexed documents",
        expanded=True,
    ):
        for document in st.session_state.indexed_documents:
            st.write(
                f'📄 **{document["filename"]}** '
                f'— {document["chunks_indexed"]} chunks'
            )


st.divider()



# Question section

st.subheader("2. Ask Your Documents")

question = st.text_area(
    "Question",
    placeholder="Example: What is covered in Session 13?",
    height=100,
)

ask_button = st.button(
    "Ask Question",
    type="primary",
    use_container_width=True,
    disabled=not question.strip() or not backend_online,
)


if ask_button:
    try:
        with st.spinner(
            "Retrieving relevant documents and generating answer..."
        ):
            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "session_id": st.session_state.session_id,
                    "question": question,
                },
                timeout=180,
            )

        if response.ok:
            result = response.json()

            st.session_state.last_result = {
                "question": question,
                "answer": result["answer"],
                "sources": result.get("sources", []),
            }

            st.session_state.messages.append(
                st.session_state.last_result
            )

        else:
            st.error(
                f"Backend error {response.status_code}: "
                f"{response.text}"
            )

    except requests.RequestException as exc:
        st.error(f"Chat request failed: {exc}")



# Current answer

if st.session_state.last_result:
    result = st.session_state.last_result

    st.divider()
    st.subheader("3. Generated Answer")

    st.info(result["answer"])

    st.subheader("4. Retrieved Documents")

    sources = result.get("sources", [])

    if sources:
        for index, source in enumerate(sources, start=1):
            with st.expander(
                f'{index}. {source["source"]} '
                f'— Page {source["page"]}',
                expanded=index == 1,
            ):
                st.write(source["snippet"])
    else:
        st.warning("No retrieved source documents were returned.")




# History

if st.session_state.messages:
    st.divider()
    st.subheader("Question History")

    for index, item in enumerate(
        reversed(st.session_state.messages),
        start=1,
    ):
        with st.expander(
            item["question"],
            expanded=False,
        ):
            st.markdown("**Generated Answer**")
            st.write(item["answer"])

            st.markdown("**Retrieved Documents**")

            for source in item.get("sources", []):
                st.write(
                    f'• {source["source"]} '
                    f'— Page {source["page"]}'
                )



# Reset

st.divider()

if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.last_result = None
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

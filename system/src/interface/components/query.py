import streamlit as st

from system.src.core.components.db.elasticsearch.retriever import ElasticsearchRetriever
from system.src.core.components.db.qdrant.retriever import QdrantRetriever
from deep_translator import GoogleTranslator

def translate(source: str, target: str, text: str):
    text = text[:4000] + "..." if len(text) > 4000 else text
    return GoogleTranslator(source=source, target=target).translate(text=text)

def render():
    st.subheader("🔍 Intelligent medical retrieval and query system")

    st.markdown("### 🧪 Sample Queries")
    st.markdown("""
    - What are the symptoms of Glaucoma?  
    - Thông tin về bệnh sốt xuất huyết?
    - What are the symptoms of High Blood Pressure?  
    - Thông tin về bệnh sỏi thận?
    - What are the symptoms of Paget's Disease of Bone?  
    - Rối loạn thính giác là gì
    - What are the symptoms of Urinary Tract Infections?  
    - Thông tin về bệnh tim mạch?
    """)

    retriever_option = st.selectbox("🧠 Choose Retriever", ["Qdrant", "Elasticsearch"])
    language_option = st.selectbox("🌐 Choose Language", ["English", "Tiếng Việt"])
    query = st.text_input("🔎 Enter your query", placeholder="e.g., What are the symptoms of Glaucoma?")
    search_clicked = st.button("🔍 Search")

    if "results" not in st.session_state:
        st.session_state.results = []
        st.session_state.current_index = 0
    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    if st.button("🧹 Clear History"):
        st.session_state.query_history = []

    if search_clicked:
        if not query:
            st.warning("⚠️ Please enter a query.")
            return

        query_to_use = translate(source="vi", target="en", text=query) if language_option == "Tiếng Việt" else query
        st.info(f"📡 Searching for: `{query_to_use}`")

        retriever = QdrantRetriever() if retriever_option == "Qdrant" else ElasticsearchRetriever()
        results = retriever.handle_query(query=query_to_use)

        if language_option == "Tiếng Việt":
            for item in results:
                answer = item["Answer"]
                max_length = 3995
                if len(answer) > max_length:
                    answer = answer[:max_length] + "..."
                    item["Anwer"] = answer
                item["Question"] = translate(source="en", target="vi", text=item["Question"])
                item["Answer"] = translate(source="en", target="vi", text=item["Answer"])

        st.session_state.results = results
        st.session_state.current_index = 0

        already_in_history = any(
            isinstance(entry, dict) and entry.get("query") == query
            for entry in st.session_state.query_history
        )
        if not already_in_history:
            st.session_state.query_history.append({
                "query": query,
                "results": results
            })

    results = st.session_state.results
    if results:
        index = st.session_state.current_index
        item = results[index]

        st.markdown(f"""
        <div style="border:1px solid #eee; border-radius: 12px; padding: 16px; background-color: #f9f9f9; margin-bottom: 16px;">
            <h4>📬 Result {index + 1} of {len(results)}</h4>
            <p><strong>🧐 Question:</strong> {item['Question']}</p>
            <p><strong>💬 Answer:</strong> {item['Answer']}</p>
            <p><strong>🎯 Score:</strong> {item['Score']:.2f}</p>
            <p>
                <div style="background-color: #e1ecf4; display: inline-block; color: #39739d; border-radius: 4px; padding: 4px 8px; font-size: 90%;">📚 Source: {item['Source']}</div>
                <div style="background-color: #fde2e2; display: inline-block; color: #d6336c; border-radius: 4px; padding: 4px 8px; font-size: 90%; margin-top: 10px;">🏷 Category: {item['Category']}</div>
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Previous") and index > 0:
                st.session_state.current_index -= 1
        with col2:
            if st.button("Next ➡️") and index < len(results) - 1:
                st.session_state.current_index += 1

    if st.session_state.query_history:
        st.markdown("---")
        st.markdown("### 🕘 Search History")

        for i, entry in enumerate(reversed(st.session_state.query_history[-10:])):
            if isinstance(entry, dict) and "query" in entry and "results" in entry:
                label = entry["query"]
                if st.button(f"🔁 {label}", key=f"history_{i}"):
                    st.session_state.results = entry["results"]
                    st.session_state.current_index = 0
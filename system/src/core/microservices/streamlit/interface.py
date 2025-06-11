import sys
import os
import streamlit as st
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# === fix import module
load_dotenv(dotenv_path="system/src/core/config/.env")

project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)


# from system.core.src.db.elasticsearch.retriever import ElasticsearchRetriever
from system.core.src.db.elasticsearch.retriever_cloud import ElasticsearchRetriever
from system.core.src.db.qdrant.retriever import ListRetriever


# ========== config for multilingual ==========
LANGUAGES = {
    "en": {
        "title": "Medical Retrieval - Q&A System",
        # "title": "Medical Retrieval System // Medical Q&A Retrieval",
        "students": "Students: Minh-Trieu, Truong & Quang-Khai, Hoang",
        "supervisor": "Supervisor: Professor Van-Dung, Hoang",
        "index": "Index",
        "id": "ID",
        "question": "Question",
        "answer": "Answer",
        "score": "Score",
        "mode_label": "Select Language"
    },
    "vi": {
        "title": "Hệ thống truy vấn, tìm kiếm và hỏi đáp tài liệu y khoa",
        "students": "Nhóm sinh viên thực hiện: Trương Minh Triều, Hoàng Quang Khải",
        "supervisor": "Giảng viên hướng dẫn: Giáo sư Hoàng Văn Dũng",
        "index": "Index",
        "id": "ID",
        "question": "Câu hỏi",
        "answer": "Câu trả lời",
        "score": "Độ chính xác",
        "mode_label": "Chọn ngôn ngữ"
    }
}


# ========== translation function ==========
def translate(text: str, src: str, dest: str):
    try:
        return GoogleTranslator(source=src, target=dest).translate(text=text)
    except:
        return text



# ========== display function ==========
def display_item(item, language):
    texts = LANGUAGES[language]
    st.markdown(f"**{texts['index']}:** {item['Index']}")
    st.markdown(f"**{texts['id']}:** {item['ID']}")

    question = item["Question"]
    answer = item["Answer"]

    if language == "vi":
        question = translate(question, "en", "vi")
        answer = translate(answer, "en", "vi")

    st.markdown(f"**{texts['question']}:** {question}")
    st.markdown(f"**{texts['answer']}:** {answer}")
    st.markdown(f"**{texts['score']}:** {item['Score']:.4f}")
    st.markdown("---")

    # st.markdown(f"**Index:** {item['Index']}")
    # st.markdown(f"**ID:** {item['ID']}")
    # st.markdown(f"**Origin question:** {item['Question']}")
    # st.markdown(f"**Origin answer:** {item['Answer']}")
    # st.markdown(f"**Score:** {item['Score']:.4f}")
    # st.markdown("---")



def main():
    st.set_page_config(layout="wide")

    language = st.selectbox(
        LANGUAGES["en"]["mode_label"],
        options=["en", "vi"],
        format_func=lambda x: "English" if x == "en" else "Tiếng Việt"
    )

    texts = LANGUAGES[language]

    st.title(texts["title"])
    # st.text("Supevisor: Van Dung Hoang")
    # st.text(texts["students"])
    st.markdown("- " + texts["supervisor"])
    st.markdown("- " + texts["students"])

    st.info("Beta version")
    st.markdown(
        """
            - Example query/ Các truy vấn mẫu cho phiên bản thử nghiệm:
            
                - English: Hearing Loss, Hearing Disorders and Deafness, Hearing Disorders
                
                - Tiếng Việt: Bệnh tim mạch, bệnh sốt xuất huyết, bệnh đau lưng, rối loạn thính giác, điếc
                
                - Lưu ý: Phiên bản hiện tại là phiên bản thử nghiệm (beta) do đó chế độ tự động phát hiện ngôn ngữ đang được đội ngũ phát triển vô hiệu hóa để có thể trải nghiệm và đánh giá chất lượng của từng loại ngôn ngữ. Vậy nên khuyến khích người dùng nên chọn đúng chế độ ngôn ngữ và thực hiện truy vấn đúng với ngôn ngữ tương tự để tăng khả năng truy vấn và độ chính xác.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        query1 = st.text_input("Search with elasticsearch:", key="col1_search")

    with col2:
        query2 = st.text_input("Search with qdrant:", key="col2_search")

    filtered1 = []
    filtered2 = []

    # logic handle query for Search with elasticsearch
    if query1:
        retriever = ElasticsearchRetriever()
        translated_query = query1
        if language == "vi":

            translated_query = translate(text=query1, src="vi", dest="en")
            print("\n\n\n\n[Search with elasticsearch] - translated_query: ", translated_query)

        filtered1 = retriever.handle_query(query=translated_query)
        
    # logic handle query for Search with qdrant
    if query2:
        retriever = ListRetriever()
        translated_query2 = query2
        if language == "vi":
            translated_query2 = translate(text=query2, src="vi", dest="en")
            print("\n\n\n\n[Search with qdrant] - translated_query: ", translated_query2)
        
        filtered2 = retriever.handle_query(query=translated_query2)

    if filtered1:
        with col1:
            for item in filtered1:
                display_item(item, language=language)

    if filtered2:
        with col2:
            for item in filtered2:
                display_item(item, language=language)


if __name__ == "__main__":
    main()

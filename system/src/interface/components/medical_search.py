import streamlit as st
import math

from system.src.interface.controllers.medical_search import MedicalSearchController

def render():
    st.subheader("🩺 Medical search")

    ITEMS_PER_PAGE = MedicalSearchController().items_per_page
    
    dataset = MedicalSearchController().dataset
    
    st.text("🔍 Search for medical informations")
    
    if "previous_query" not in st.session_state:
        st.session_state.previous_query = ""
        
    query = st.text_input("Enter disease name", "")
    
    if query != st.session_state.previous_query:
        st.session_state.current_page = 1
        st.session_state.previous_query = query
        
    all_medicals = MedicalSearchController().get_medical_list()
    filtered_medicals = [med for med in all_medicals if query.lower() in med.lower()]
    total_items = len(filtered_medicals)
    total_pages = max(math.ceil(total_items / ITEMS_PER_PAGE), 1)

    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    start_idx = (st.session_state.current_page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    current_page_meds = filtered_medicals[start_idx:end_idx]

    selected_medical = None
    if current_page_meds:
        st.markdown("### 📋 List of diseases:")
        cols = st.columns(3)
        for idx, med in enumerate(current_page_meds):
            with cols[idx % 3]:
                if st.button(med, key=f"{st.session_state.current_page}-{med}"):
                    selected_medical = med
    else:
        st.info("No informations found.")

    
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        if st.button("⬅️ previous") and st.session_state.current_page > 1:
            st.session_state.current_page -= 1
            st.rerun()

    with col2:
        st.markdown("**Previous page:**")
        st.markdown("`1`")

    with col3:
        st.markdown("**Current page / total:**")
        st.markdown(f"`{st.session_state.current_page}` / `{total_pages}`")

    with col4:
        st.markdown("**Last page:**")
        st.markdown(f"`{total_pages}`")

    with col5:
        if st.button("Next ➡️") and st.session_state.current_page < total_pages:
            st.session_state.current_page += 1
            st.rerun()

    if selected_medical:
        st.markdown("---")
        st.markdown(f"### 🧾 Medical information: `{selected_medical}`")
        questions = MedicalSearchController().get_questions_for_disease(selected_medical)

        if questions:
            st.write("Information found:")

            for q in questions:
                with st.expander(f"🔸 {q}"):
                    answers = dataset[dataset["question"] == q]["answer"].values.tolist()
                    if answers:
                        best_answer = sorted(answers, key=lambda x: len(x), reverse=True)[0]
                        st.markdown(best_answer)
                    else:
                        st.warning("No answer found for this question.")
        else:
            st.info("No information found for this drug.")
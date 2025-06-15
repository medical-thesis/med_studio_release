import streamlit as st
import math

from system.src.interface.controllers.drug_discovery import DrugDiscoveryController

def render():
    st.subheader("💊 Drug discovery")

    ITEMS_PER_PAGE = DrugDiscoveryController().items_per_page
    
    st.text("🔍 Search for drug information")
    
    if "previous_query" not in st.session_state:
        st.session_state.previous_query = ""
        
    query = st.text_input("Enter drug name", "")
    
    if query != st.session_state.previous_query:
        st.session_state.current_page = 1
        st.session_state.previous_query = query

    all_medicines = DrugDiscoveryController().get_medicine_list()
    filtered_medicines = [med for med in all_medicines if query.lower() in med.lower()]
    total_items = len(filtered_medicines)
    total_pages = max(math.ceil(total_items / ITEMS_PER_PAGE), 1)

    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    start_idx = (st.session_state.current_page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    current_page_meds = filtered_medicines[start_idx:end_idx]

    selected_medicine = None
    if current_page_meds:
        st.markdown("### 📋 List of drugs:")
        cols = st.columns(3)
        for idx, med in enumerate(current_page_meds):
            with cols[idx % 3]:
                if st.button(med, key=f"{st.session_state.current_page}-{med}"):
                    selected_medicine = med
    else:
        st.info("No drugs found.")

    
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

    if selected_medicine:
        st.markdown("---")
        st.markdown(f"### 🧾 Drug information: `{selected_medicine}`")
        st.markdown(DrugDiscoveryController().load_medicine_info(selected_medicine))
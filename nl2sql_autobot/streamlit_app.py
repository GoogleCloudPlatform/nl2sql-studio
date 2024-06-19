from bot_functions import *
from nl2sql import *
import streamlit as st
import time

def set_page_layout():
    st.set_page_config(
        page_title="NL2SQL Autobot",
        layout="wide",
    )

def show_past_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"].replace("$", r"\$"))  # noqa: W605
            try:
                if message["backend_details"]:
                    with st.expander("Intermediate Steps:"):
                        st.markdown(message["backend_details"])
            except KeyError:
                pass


if __name__ == '__main__':
    set_page_layout()
    if "chat" not in st.session_state:
        st.session_state.chat = model.start_chat()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([7,1])
    with col1:
        st.title("Autonomus NL2SQL Bot for your Database")
    with col2:
        clear_button = st.button("New Chat", help="Deletes the conversation history, which makes your new chat faster", type="primary", use_container_width=True)


    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Click for Sample Questions", expanded=False):
            st.write("""
                - What kind of information is in this database?
                - Tell about all the tables in this DB.
                - how many contracts expired last year?
                - What agreements are going to expire in next 5 years?
                - what time period of data exists in the agreement table
                - Show me a chart of most common months of effective_end_date
                - Show me a chart of the years and how many contracts expire that year
                - give me interesting ideas of plots whih can be drawn on this database
            """,)
    with col2:
        with st.expander("Click to Configure your DB searches"):
            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                project_id = st.text_input(label='Project id', value=PROJECT_ID)
            with col2:
                dataset_id = st.text_input(label='Dataset id', value=DATASET_ID)
            with col3:
                tables_lit = st.text_input(label='List of tables to analyze', value=', '.join(TABLES_LIST))


    prompt = st.chat_input("Ask me about information in the database... Make sure to clear conversation to ask a question on new topic.")

    if clear_button:
        st.session_state.chat = model.start_chat()
        st.session_state.messages = []

    show_past_chat()
    print(len(st.session_state.chat.history))

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            with st.spinner('I am working. Please wait...'):
                response = ask(prompt, st.session_state.chat)
            # time.sleep(1)

            full_response = response.text
            interims = format_interim_steps(response.interim_steps)
            with message_placeholder.container():
                st.markdown(full_response.replace("$", r"\$"))  # noqa: W605
                if interims:
                    with st.expander("Intermediate Steps:"):
                        st.markdown(interims)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "backend_details": interims,
        })






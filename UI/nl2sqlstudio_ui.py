# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    File describing the layout of the User Interface
    using Streamlit library widgets
    Streamlit entry point function
"""
import os
import json
from io import StringIO
import requests
from dotenv import load_dotenv
from loguru import logger

import streamlit as st
from streamlit_modal import Modal

# import the executors
from utils import default_func, linear_gen_sql, cot_gen_sql, rag_gen_sql
# import support functions
from utils import get_feedback, message_queue, add_question_to_db
# import auth functions
from utils import view_auth_google, view_login_google, back_to_login_page

load_dotenv()
SHOW_SUCCESS = False
st.set_page_config(page_title='NL2SQL Studio',
                   page_icon="📊",
                   initial_sidebar_state="expanded",
                   layout='wide')


def define_session_variables():
    """
        Define the session variables once at the start of the app
    """
    logger.info("Defining Session variables")
    st.session_state.messages = []
    st.session_state.question = ''
    st.session_state.new_question = False
    st.session_state.user_response = 0
    st.session_state.user_responded = False
    st.session_state.fb_count = 1
    st.session_state.refresh = True

    st.session_state.add_sample_question = False
    st.session_state.sample_question = ''
    st.session_state.sample_sql = ''
    st.session_state.add_question_status = False
    st.session_state.result_id = ''

    # st.session_state.access_token = None
    # st.session_state.token = None
    # st.session_state.login_status = False


def define_modals():
    """
        Define the Modal dialogs for Help Info display, Sample Queries
        and Project Configuration
    """
    q_s_modal = Modal(
            "Query Selector",
            key="qsm",
            # Optional
            padding=10,    # default value
            max_width=700  # default value
        )
    pc_modal = Modal('Project Configuration',
                     key='pcm',
                     padding=10,
                     max_width=545)
    qa_modal = Modal('Sample QnA',
                     key='qam',
                     padding=10,
                     max_width=545)
    info_modal = Modal('Success !!',
                       key='info',
                       padding=2,
                       max_width=200)

    st.session_state.pc_modal = pc_modal
    st.session_state.qa_modal = qa_modal
    st.session_state.info_modal = info_modal
    st.session_state.q_s_modal = q_s_modal


def define_pre_auth_layout():
    """
        Define the Login page prior to Login
    """
    def logo_image():
        """
            Display the Logo image and the Login button
        """

        left_co, cent_co, last_co = st.columns([0.35, 0.35, 0.3])
        with cent_co:
            st.image('solid_g-logo-2.png', )
            lc, cc, rc = st.columns([0.3, 0.4, 0.3])
            with cc:
                login_link()

        # st.image('solid_g-logo-2.png')

    def login_link():
        """
            Design and Link for the Login button
            called in logo_image function
        """
        st.markdown("""
            <style>
            .big-font {
                font-size:20px !important;
                background-color: royalblue;
                border-radius: 20%/50%;
                width: 105px;
                height: 1.75em;
                padding: 0px 0px 0px 30px ;
            }
            </style>
            """, unsafe_allow_html=True)
        auth_url = view_login_google()
        hyperlink_string = '<div class="big-font"> <a href="' + auth_url + '" style="color:white" target="_self">Login</a> </div>'
        logger.info(f"Auth url = {auth_url}")
        st.markdown(hyperlink_string, unsafe_allow_html=True)

    logo_image()


def define_post_auth_layout():
    """
        Streamlit UI layout with page configuration, styles,
        widgets, main screen and sidebar, etc
    """
    def page_config():
        st.set_page_config(page_title='NL2SQL Studio',
                           page_icon="📊",
                           initial_sidebar_state="expanded",
                           layout='wide')

    def markdown_styles():
        st.markdown("""
            <style>
                .block-container {
                        padding-top: 0.25rem;
                        padding-bottom: 0rem;
                        padding-left: 5rem;
                        padding-right: 5rem;
                    }
            </style>
            """, unsafe_allow_html=True)

        st.markdown(
            """
            <style>
                [data-testid=stImage]{
                    text-align: center;
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 100%;
                }
            </style>
            """, unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
                [data-testid=stSidebar] [data-testid=stImage]{
                    text-align: center;
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 100%;
                }
            </style>
            """, unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
                [data-testid=stContainer] [data-testid=stImage]{
                    text-align: center;
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 100%;
                    border: 2px;
                    min-height: 30%;
                    max-height: 50%;
                }
            </style>
            """, unsafe_allow_html=True
        )

    # Side Bar definintion

    def sidebar_components():
        """
            UI Controls in the Sidebar panel
        """
        with st.sidebar.container():
            column_1, column_2 = st.columns(2)
            with column_1:
                st.image('google.png')
            with column_2:
                st.write('v0.4')
                logout_state = st.button("Logout")

        with st.sidebar.container(height=140):
            st.session_state.model = st.radio('Pick your Model',
                                              ['Linear Executor',
                                               'Rag Executor',
                                               'Chain of Thought'])
        with st.sidebar.expander("Configuration Settings"):
            proj_conf = st.button("Project Configuration")
            rag_input = st.button("Questions  &  Queries", disabled=False)

        with st.sidebar.container(height=60):
            st.session_state.execution = st.checkbox(
                "Generate and Execute",
                disabled=False
                )

        if proj_conf:
            pc_modal = st.session_state.pc_modal
            pc_modal.open()

        if rag_input:
            qa_modal = st.session_state.qa_modal
            qa_modal.open()

        if logout_state:
            logger.info("Logging out")
            st.session_state.token = None
            st.session_state.login_status = False
            back_to_login_page()
            st.query_params.clear()

    def main_page():
        """
            UI Controls and interaction on the Main panel
        """
        def logo_image():
            st.image('solid_g-logo-2.png')

        def help_info():
            with st.container():
                column_1, column_2, column_3 = st.columns([0.25, 0.85, 0.1])
                with column_1:
                    # Nothing to do. Leave blank
                    pass
                with column_2:
                    # Nothing to do. Leave blank
                    pass
                with column_3:
                    st.markdown('',
                                help="""For the purpose of this demo we have
                                setup a demo project with id
                                'sl-test-project-363109' created a dataset in
                                BigQuery named 'zoominfo'. This dataset
                                contains 3 tables with information that is a
                                subset of Zoominfo Data Cubes. This a the
                                default dataset to generate SQLs from related
                                natural language statements.  For custom query
                                generation, specify the Project ID, Dataset and
                                Metadata of tables in the Configuration
                                settings in the Sidebar panel""")

        def input_container():
            """
                Define the Question input entry and Sample queries button
            """
            inp_container = st.container()
            with inp_container:
                column_1, column_2 = st.columns([0.86, 0.14])
                with column_2:
                    q_s = st.button('Sample Queries', key='qs_button')
                with column_1:
                    if question := st.chat_input("Enter your question here"):
                        message_queue(question)
                        st.session_state.question = question
                        st.session_state.new_question = True
                        st.session_state.user_responded = False
            st.session_state.ic = inp_container
            if q_s:
                q_s_modal = st.session_state.q_s_modal
                q_s_modal.open()

        def qa_msgs_container():
            """
                Main chat session window of the screen
            """
            msg_container_main = st.container(height=425)
            with msg_container_main:
                column_1, column_2 = st.columns([0.90, 0.10])
                with column_1:
                    msg_container = st.container()
                with column_2:
                    fb_container = st.container()

            st.session_state.fc = fb_container
            st.session_state.mc = msg_container
            get_feedback()

        def disclaimer():
            """
                Disclaimer message at the bottom of the screen
            """
            st.markdown("<p style='text-align: center; font-style: italic;\
                        font-size: 0.75rem;'>\
                        The SQL generated by this tool may be inaccurate\
                        or incomplete. Always review and test the code before\
                        executing it against your database.</p>",
                        unsafe_allow_html=True)

        def sample_queries_modal_active():
            """
                Modal output when the Sample queries button is pressed
            """
            q_s_modal = st.session_state.q_s_modal
            if q_s_modal.is_open():
                with open('sample_questions.txt',
                          'r',
                          encoding="utf-8") as input_file:
                    questions_list = input_file.readlines()

                with q_s_modal.container():
                    # st.title("Copy any sample question")
                    for question in questions_list:
                        st.code(question)

                    if st.button("Close"):
                        q_s_modal.close()

        def qa_modal_active():
            """
                Modal output when the Questions and Queries button
                on the Side bar panel is pressed
            """
            qa_modal = st.session_state.qa_modal
            if qa_modal.is_open():
                with qa_modal.container():
                    samp_question = st.text_input('Enter sample question')
                    samp_sql = st.text_input(("Enter corresponding SQL"))
                    if st.session_state.add_question_status:
                        st.success("Success ! Question added to DB ")
                    if st.button('Add question'):
                        add_question_to_db(samp_question, samp_sql)
                        info_modal = st.session_state.info_modal
                        info_modal.open()
                        qa_modal.close(True)

        def pc_modal_active():
            """
                Modal output when the Project Configuration button on the
                Side bar is pressed
            """
            pc_modal = st.session_state.pc_modal
            if pc_modal.is_open():
                with pc_modal.container():
                    project = st.text_input('Mention the GCP project name')
                    dataset = st.text_input(
                        'Specify the BigQuery dataset name'
                        )
                    uploaded_file = st.file_uploader(
                        "Choose the Metadata Cache file"
                        )
                    if st.button("Save configuration"):
                        if uploaded_file is not None:
                            # To read file as bytes:
                            url = os.getenv('EXECUTORS')
                            # To convert to a string based IO:
                            stringio = StringIO(
                                uploaded_file.getvalue().decode("utf-8")
                                )

                            # To read file as string:
                            string_data = stringio.read()
                            files = {'file': (uploaded_file.name, string_data)}
                            body = {"proj_name": project,
                                    "dataset": dataset,
                                    "metadata_file": uploaded_file.name}
                            headers = {"Content-type": "application/json",
                                       "Authorization":
                                       f"Bearer {st.session_state.access_token}"}
                            # url = "http://localhost:5000"
                            _ = requests.post(url=url+"/projconfig",
                                              data=json.dumps(body),
                                              headers=headers,
                                              timeout=None)
                            _ = requests.post(url=url+"/uploadfile",
                                              headers={"Authorization":
                                                       f"Bearer {st.session_state.access_token}"},
                                              files=files,
                                              timeout=None)

                        pc_modal.close()

        # Main page function calls
        main_page_functions = {
            "logo_image": logo_image,
            "help_info": help_info,
            "input_container": input_container,
            "qs_msgs_container": qa_msgs_container,
            "disclaier": disclaimer,
            "sample_queries_modal": sample_queries_modal_active,
            "qa_modal": qa_modal_active,
            "pc_modal": pc_modal_active
        }

        for _, mp_function in main_page_functions.items():
            mp_function()

        st.session_state.add_question_status = False

    # Layout function calls
    layout_functions = {
            # "page_config": page_config,
            "markdown_styles": markdown_styles,
            "sidebar_components": sidebar_components,
            "main_page": main_page
        }

    for _, layout_function in layout_functions.items():
        layout_function()


def pre_initialize():
    """
        Initialise the Application context
    """

    if 'init' not in st.session_state:
        define_session_variables()
        st.session_state.init = False

    logger.info(f"Login status = {st.session_state.login_status}")


def initialize():
    """
        Initialise the Application context
    """
    if 'init' not in st.session_state:
        define_session_variables()
        st.session_state.init = False

    define_modals()
    define_post_auth_layout()

    if "messages" not in st.session_state:
        st.session_state.messages = []


def redraw():
    """
        Trigger the re-rendering of the UI
    """
    # st.mc.empty()
    msg_container = st.session_state.mc
    with msg_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def add_new_question():
    """
        Function that is called when a new question is added to the
        messae queue.  This will trigger the API calls to invoke the
        appropriate Executor that is selected on the Sidebar panel
    """
    if 'new_question' in st.session_state:
        redraw()
        st.session_state.refresh = False
        if st.session_state.new_question:
            if st.session_state.model == 'None':
                default_func(st.session_state.question)
            elif st.session_state.model == 'Linear Executor':
                linear_gen_sql(st.session_state.question)
            elif st.session_state.model == 'Chain of Thought':
                cot_gen_sql(st.session_state.question)
            else:
                rag_gen_sql(st.session_state.question)


def when_user_responded():
    """
        Function to capture the user feedback from the
        Thumbs up/down widget
    """
    if st.session_state.user_responded:
        st.session_state.user_responded = False
        resp = st.session_state.messages[-1]['content']
        user_feedback = 'True'\
            if st.session_state.user_response == 1 else 'False'
        if user_feedback == 'True':
            info_text = ':green[👍 User feedback captured ]'
        else:
            info_text = ':red[👎 User feedback captured ]'

        st.session_state.messages[-1]['content'] = resp + " \n\n" + info_text
        url = os.getenv('EXECUTORS') + '/userfb'
        data = {"result_id": st.session_state.result_id,
                "user_feedback": user_feedback}

        logger.info(f"User reposnse data to API {data}")
        headers = {'Content-type': 'application/json',
                   'Accept': 'text/plain',
                   "Authorization": f"Bearer {st.session_state.access_token}"}
        resp = requests.post(url=url,
                             data=json.dumps(data),
                             headers=headers,
                             timeout=None)

        st.session_state.refresh = True
        get_feedback()


def refresh():
    """
        Refresh the display
    """
    if st.session_state.refresh:
        # redraw()
        st.rerun()
    else:
        st.session_state.refresh = True
        # st.rerun()


def app_load():
    """
        On Application load
    """
    logger.info("App loaders")
    found_query_params = False
    try:
        logger.info(f"Query Parameters - {st.query_params}")
        code = st.query_params['code']
        found_query_params = True
    except Exception:
        logger.info("Login required")
        found_query_params = False

    if found_query_params:
        id_token, access_token = view_auth_google(st.query_params['code'])
        logger.info(f"ID Token = {id_token}")
        logger.info(f"Access Token = {access_token}")
        st.session_state.token = id_token
        st.session_state.access_token = access_token
        st.session_state.login_status = True
    else:
        st.session_state.token = None
        st.session_state.access_token = None
        st.session_state.login_status = False

    logger.info(f"Login status = {st.session_state.login_status}")


def render_view():
    pre_auth_post_logout = {
        "pre-init": pre_initialize,
        # "app_load": app_load,
        "pre_auth_page": define_pre_auth_layout,
    }

    post_auth = {
        # "app_load": app_load,
        "initialize": initialize,
        # 'modals': define_modals,
        # "layout" : define_post_auth_layout,
        # "redraw": redraw,
        "add_new_question": add_new_question,
        "when_user_responded": when_user_responded,
        "refresh": refresh
    }

    app_load()
    funcs_to_exec = post_auth if st.session_state.login_status \
        else pre_auth_post_logout

    for _, function in funcs_to_exec.items():
        function()

render_view()

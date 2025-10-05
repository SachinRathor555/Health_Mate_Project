GOOGLE_API_KEY = "YOUR_GEMINI_API"

#healthmate.db
#https://github.com/Jnan-py/HealthMate/blob/main/healthmate.db #

import os
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
import google.generativeai as genai
from pypdf import PdfReader
import hashlib
import sqlite3


### SQLITE3 things
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

UPLOAD_DIR = "user_uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

DB_NAME = "healthmate.db"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        #Users table
    with sqlite3.connect(DB_NAME) as conn:        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
@@ -30,7 +31,6 @@ def init_db():
        )
        """)

        # Files Table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
@@ -79,22 +79,55 @@ def get_user_files(user_id):
        """, (user_id,)).fetchall()
        return files

init_db()

#### GEMINI API Things

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

import google.generativeai as genai
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
def delete_file(user_id, file_name):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
        DELETE FROM files WHERE (file_name, user_id) = (?, ?)
        """, (file_name, user_id)).fetchall()
        conn.commit()

def get_gemini_response(prev_chat):
    model = genai.GenerativeModel(model_name='gemini-pro')
init_db()

    response = model.generate_content(f''' 
    Prompt:
embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 200)
    chunks = splitter.split_text(text)
    return chunks

@st.cache_resource
def get_vector_store(chunks):
    db = FAISS.from_texts(chunks, embedding = embeddings)
    return db

def get_rel_text(user_query,db):
    rel_text = db.similarity_search(user_query, k = 1)
    return rel_text[0].page_content
 
def bot_response(model, query, relevant_texts, history): 
    context = ' '.join(relevant_texts)
    prompt = f"""This is the context of the document 
    Context: {relevant_texts}
    And this is the user query
    User: {query}
    And this is the history of the conversation
    History: {history}
    Please generate a response to the user query based on the context and history
    The questions might be asked related to the provided context, and may also be in terms of the medical terms, diseases, etc..,
    Answer the query with respect to the medical report context provided, you can also use your additional knowledge too, but do not ignore the content of the provided medical report,
    Answer the queries like a professional doctor, having a lot of knowledge on the based report context
    Bot:
    """
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.65,
        )
    )
    return response.text

model = genai.GenerativeModel(model_name='gemini-2.0-flash', 
    system_instruction = """
    Your name is "CuraBot" and you are a doctor who gives the medications and let the user know the disease he is suffering from based on the symptoms he provides
    
        Your Role:
@@ -112,163 +145,164 @@ def get_gemini_response(prev_chat):
    
    This is so important and I want you to stick to these points everytime without any mismatches, and I want you to maintain the consistency too.
    First start with the greetings message like "Welcome, How can I help you with the diagnosis today..??"
    """
    )

st.set_page_config(page_title="HealthMate", page_icon="🩺", layout = "wide")

if 'messages' not in st.session_state:
    st.session_state.messages = {}

with st.sidebar:
    selected = option_menu(
        "Menu", ["Landing Page","Login / SignUp","Consultation", "Medical Record Reader"],
        icons=["house", None,"chat", "file-medical"],
        menu_icon="cast", default_index=0
    )

if selected == 'Landing Page':
    st.title("HealthMate")
    st.header('Where Health Diagnosis Meets Technology')
    st.markdown("""
    In today’s fast-paced world, prioritizing your health and managing medical records shouldn’t be a hassle. 
    That’s where **HealthMate** steps in, revolutionizing how you approach healthcare. With HealthMate, you gain access to two powerful tools designed to simplify and enhance your healthcare journey:
    - **Symptom Checker and Medication Advisor Chatbot**
    - **Medical Record Reader and Organizer**
    """)

    st.subheader("🩺 Symptom Checker and Medication Advisor")
    st.markdown("""
    Not feeling well? Wondering what those symptoms could mean? 
    The **Symptom Checker and Medication Advisor Chatbot** is here to assist you anytime, anywhere.

    ### **Features:**
    - **24/7 Symptom Analysis:** Describe your symptoms and receive instant insights.
    - **Personalized Recommendations:** Get advice on medications and remedies tailored to your needs.
    - **Lifestyle Tips:** Learn practical steps to enhance your health.
    - **Medical Advice:** Know when it’s time to consult a doctor.

    ### **How It Works:**
    1. Start a chat and describe your symptoms.
    2. Let the AI-powered chatbot analyze your input.
    3. Receive personalized recommendations and next steps.

    """)

    st.subheader("📂 Medical Record Reader and Organizer")
    st.markdown("""
    Managing medical records can often feel overwhelming. With HealthMate's **Medical Record Reader and Organizer**, 
    you can easily upload, store, and access your health documents at the click of a button.

    ### **Features:**
    - **Secure Uploads:** Safely upload medical records from your device.
    - **Easy Access:** Retrieve documents anytime, anywhere.

    ### **How It Works:**
    1. Upload your medical records using the secure uploader.
    2. Let the app organize and analyze your records.
    3. Access or share your records as needed.

    """)

if selected == 'Login / SignUp':
    st.header("Login or Sign Up")

    if "user_id" in st.session_state:
        st.info(f"You are logged in as {st.session_state['first_name']} {st.session_state['last_name']}.")
        if st.button("Log Out"):
            st.session_state.clear()
            st.success("Logged out successfully!")

    else:
        action = st.selectbox("Select an action", ["Login", "Sign Up"])  

        if action == "Sign Up":
            st.subheader("Sign Up")
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            dob = st.date_input("Date of Birth")  
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Create Account"):
                success, msg = sign_up(first_name, last_name, dob, email, password)
                if success:
                    st.success(str(msg))  
                else:
                    st.error(str(msg))

        elif action == "Login":
            st.subheader("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Log In"):
                user = login(email, password)
                if user: 
                    st.session_state["user_id"], st.session_state["first_name"], st.session_state["last_name"] = user
                    st.success(f"Logged in as {user[1]} {user[2]}!")
                    st.session_state.messages[st.session_state['user_id']]  = []
                else:
                    st.error("Invalid email or password.")

if selected == "Consultation":
    st.title("Chat with HealthMate")
    if 'user_id' not in st.session_state:
        st.warning('You need to login first')

    else:
        st.info(f'Welcome {st.session_state['first_name']} {st.session_state['last_name']} !!')
        st.write("Describe your symptoms, ask for a diagnosis, or simply say hello!")
        chat_history = st.session_state.messages.get(st.session_state['user_id'], [])

        chat_bot = model.start_chat(
                        history = chat_history,
                    )

        for message in chat_history:
            row = st.columns(2)
            if message['role']=='user':
                row[1].chat_message(message['role']).markdown(message['parts'])
            else:
                row[0].chat_message(message['role']).markdown(message['parts'])

The previous chat is provided, if the previous chat is not provided then consider that the session just started and greet the user and wait for his response
        Previous Chat : {prev_chat}
    ''')

    content = response.text
    return content


### STREAMLIT UI    

def main():
    st.set_page_config(page_title="HealthMate", page_icon="🩺")

    with st.sidebar:
        selected = option_menu(
            "Menu", ["Landing Page","Login / SignUp","Consultation", "Medical Record Reader"],
            icons=["house", None,"chat", "file-medical"],
            menu_icon="cast", default_index=0
        )

    if selected == 'Landing Page':
        st.title("HealthMate")
        st.header('Where Health Diagnosis Meets Technology')
        st.markdown("""
        In today’s fast-paced world, prioritizing your health and managing medical records shouldn’t be a hassle. 
        That’s where **HealthMate** steps in, revolutionizing how you approach healthcare. With HealthMate, you gain access to two powerful tools designed to simplify and enhance your healthcare journey:
        - **Symptom Checker and Medication Advisor Chatbot**
        - **Medical Record Reader and Organizer**
        """)

        st.subheader("🩺 Symptom Checker and Medication Advisor")
        st.markdown("""
        Not feeling well? Wondering what those symptoms could mean? 
        The **Symptom Checker and Medication Advisor Chatbot** is here to assist you anytime, anywhere.

        ### **Features:**
        - **24/7 Symptom Analysis:** Describe your symptoms and receive instant insights.
        - **Personalized Recommendations:** Get advice on medications and remedies tailored to your needs.
        - **Lifestyle Tips:** Learn practical steps to enhance your health.
        - **Medical Advice:** Know when it’s time to consult a doctor.

        ### **How It Works:**
        1. Start a chat and describe your symptoms.
        2. Let the AI-powered chatbot analyze your input.
        3. Receive personalized recommendations and next steps.

        """)

        st.subheader("📂 Medical Record Reader and Organizer")
        st.markdown("""
        Managing medical records can often feel overwhelming. With HealthMate's **Medical Record Reader and Organizer**, 
        you can easily upload, store, and access your health documents at the click of a button.

        ### **Features:**
        - **Secure Uploads:** Safely upload medical records from your device.
        - **Easy Access:** Retrieve documents anytime, anywhere.

        ### **How It Works:**
        1. Upload your medical records using the secure uploader.
        2. Let the app organize and analyze your records.
        3. Access or share your records as needed.

        """)

    if selected == 'Login / SignUp':
        st.header("Login or Sign Up")

        if "user_id" in st.session_state:
            st.info(f"You are logged in as {st.session_state['first_name']} {st.session_state['last_name']}.")
            if st.button("Log Out"):
                st.session_state.clear()
                st.success("Logged out successfully!")

        else:
            action = st.selectbox("Select an action", ["Login", "Sign Up"])  

            if action == "Sign Up":
                st.subheader("Sign Up")
                first_name = st.text_input("First Name")
                last_name = st.text_input("Last Name")
                dob = st.date_input("Date of Birth")  
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.button("Create Account"):
                    success, msg = sign_up(first_name, last_name, dob, email, password)
                    if success:
                        st.success(str(msg))  
                    else:
                        st.error(str(msg))

            elif action == "Login":
                st.subheader("Login")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.button("Log In"):
                    user = login(email, password)
                    if user: 
                        st.session_state["user_id"], st.session_state["first_name"], st.session_state["last_name"] = user
                        st.success(f"Logged in as {user[1]} {user[2]}!")
                    else:
                        st.error("Invalid email or password.")


    if selected == "Consultation":
        if 'user_id' not in st.session_state:
            st.warning('You need to login first')
        user_question = st.chat_input("Enter your symptoms here !!")

        else:
            st.title("Chat with HealthMate")
            st.info(f"Welcome {st.session_state['first_name']} {st.session_state['last_name']} !!")
        if user_question:
            row_u = st.columns(2)
            row_u[1].chat_message('user').markdown(user_question)
            chat_history.append(
                {'role':'user',
                'parts':user_question}
            )

            st.write("Describe your symptoms, ask for a diagnosis, or simply say hello!")
            with st.spinner("Thinking ..."):
                response = chat_bot.send_message(user_question)

            if 'messages' not in st.session_state:
                st.session_state.messages = []
                row_a = st.columns(2)
                row_a[0].chat_message('assistant').markdown(response.text)

                st.session_state.messages.append(
                    {
                        'role':'assistant',
                        'content':'Welcome !! Start listing your symptoms and get the accurate diagnosis'
                    }
                chat_history.append(
                    {'role':'assistant',
                    'parts':response.text}
                )

            for message in st.session_state.messages:
                row = st.columns(2)
                if message['role']=='user':
                    row[1].chat_message(message['role']).markdown(message['content'])
                else:
                    row[0].chat_message(message['role']).markdown(message['content'])
            st.session_state.messages[st.session_state['user_id']] = chat_history

            user_question = st.chat_input("Enter your symptoms here !!")
        
            if user_question:
                row_u = st.columns(2)
                row_u[1].chat_message('user').markdown(user_question)
                st.session_state.messages.append(
                    {'role':'user',
                    'content':user_question}
                )
elif selected == "Medical Record Reader":
    st.title("Medical Record Reader")

                response = get_gemini_response(user_question)
    if 'user_id' not in st.session_state:
        st.warning('You need to login first')
    
    else:
        with st.expander("Select the feature ", expanded = True):
            choice = st.radio(
                label = "Select the type",
                options = ["Upload the Medical Record", "Chat with Medical Record"]
            )

                row_a = st.columns(2)
                row_a[0].chat_message('assistant').markdown(response)
                st.session_state.messages.append(
                    {'role':'assistant',
                    'content':response}
                )
        st.info(f'Welcome {st.session_state['first_name']} {st.session_state['last_name']} !!')

    elif selected == "Medical Record Reader":
        if 'user_id' not in st.session_state:
            st.warning('You need to login first')
        
        else:
            st.title("Medical Record Reader")
            st.info(f'Welcome {st.session_state['first_name']} {st.session_state['last_name']} !!')
        if choice == "Upload the Medical Record":
            file = st.file_uploader(label='Upload your medical record',type='pdf')

            if file:
@@ -286,6 +320,8 @@ def main():
            if files:
                for file_name, file_path in files:
                    st.markdown(f"- {file_name}")
                    if st.button(f"Delete {file_name}"):
                        delete_file(st.session_state["user_id"], file_name)

                st.subheader('File Content Viewer')
                s_file = st.selectbox(label='Select the file', options=[i for i,v in files])
@@ -300,6 +336,7 @@ def get_value(i, lst):
                    file_path = get_value(s_file,files)
                    if st.button('View Content'):
                        with st.spinner('Giving the details'):
                            
                            pdf_reader = PdfReader(file_path)
                            text = ''
                            for page in pdf_reader.pages:
@@ -309,8 +346,90 @@ def get_value(i, lst):
                            st.write(text)
            else:
                st.info("No files uploaded yet.")
        
        elif choice == "Chat with Medical Record":
            if "doc_paragraphs" not in st.session_state:
                st.session_state.doc_paragraphs = {}
            if "doc_messages" not in st.session_state:
                st.session_state.doc_messages = {}
            if "faiss" not in st.session_state:
                st.session_state.faiss = {}

            st.subheader("Chat with Medical Record")
            files = get_user_files(st.session_state["user_id"])
            s_file = st.selectbox(label='Select the file', options=[i for i,v in files])

            def get_value(i, lst):
                for pair in lst:
                    if pair[0] == i:  
                        return pair[1]  
                return None

            if s_file:
                if s_file not in st.session_state.doc_messages:
                    st.session_state.doc_messages[s_file] = []

                file_path = get_value(s_file,files)

                if s_file not in st.session_state.doc_paragraphs:
                    with st.spinner('Getting the details'):
                        pdf_reader = PdfReader(file_path)
                        text = ''
                        for page in pdf_reader.pages:
                            text+= page.extract_text()
                    
                        st.session_state.doc_paragraphs[s_file] = text
                    

                if s_file not in st.session_state.faiss:
                    chunks = get_chunks(st.session_state.doc_paragraphs[s_file])

                    with st.spinner("Reading records..."):
                        st.session_state.faiss[s_file] = get_vector_store(chunks)

                h_model = genai.GenerativeModel(model_name= "gemini-2.0-flash", 
                system_instruction = "You are a very professional doctor with a lots of years of experience and you are here to help the patient with their medical record."
                )

                doc_chat = st.session_state.doc_messages.get(s_file)

if __name__ == "__main__":
    main()
                for message in doc_chat:
                    row = st.columns(2)
                    if message['role'] == 'user':
                        row[1].chat_message(message['role']).markdown(message['content'])
                    else:
                        row[0].chat_message(message['role']).markdown(message['content'])

                try:
                    user_question = st.chat_input("Enter your query here !!")

                    if user_question:
                        row_u = st.columns(2)
                        row_u[1].chat_message('user').markdown(user_question)
                        doc_chat.append(
                            {'role': 'user',
                            'content': user_question}
                        )

                        with st.spinner("Generating response..."):
                            relevant_texts = get_rel_text(user_question, st.session_state.faiss[s_file])
                            bot_reply = bot_response(h_model, user_question, relevant_texts, doc_chat)

                        row_a = st.columns(2)
                        row_a[0].chat_message('assistant').markdown(bot_reply)

                        doc_chat.append(
                            {'role': 'assistant',
                            'content': bot_reply}
                        )


                except Exception as e:
                    st.chat_message('assistant').markdown(f'There might be an error, try again, {str(e)}')
                    doc_chat.append(
                        {
                            'role': 'assistant',
                            'content': f'There might be an error, try again, {str(e)}'
                        }
                    )

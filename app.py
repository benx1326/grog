import streamlit as st

import requests
import json
import time
import os


def stream_output(s):
    for line in s.split('\n'):
        for word in line.split(' ')[1:]:
            yield f"{word} "
            time.sleep(0.01)
        yield f"\n"
def query_flask_app(prompt):
    url = 'http://localhost:8000/query'  # URL of your Flask app endpoint
    headers = {'Content-Type': 'application/json'}
    data = {'prompt': prompt}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        return response_data.get('response'), response_data.get('relevant_docs', [])
    else:
        print(f"Error: {response.status_code}")
        return None, []

st.title("TinyML Foundation RAG Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []
if 'init' not in st.session_state:
    os.system('python server.py &') 
    st.session_state.init = True

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        prompt = f'{prompt}?' if prompt[-1] != '?' else prompt
        stream, relevant_docs = query_flask_app(prompt)
        response = st.write_stream(stream_output(stream))
        st.session_state.messages.append({"role": "assistant", "content": response})


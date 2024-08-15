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


if 'search_engine' not in st.session_state:
    import os

    import pandas as pd
    import tiktoken

    from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
    from graphrag.query.indexer_adapters import (
        read_indexer_covariates,
        read_indexer_entities,
        read_indexer_relationships,
        read_indexer_reports,
        read_indexer_text_units,
    )
    from graphrag.query.input.loaders.dfs import (
        store_entity_semantic_embeddings,
    )
    from graphrag.query.llm.oai.chat_openai import ChatOpenAI
    from graphrag.query.llm.oai.embedding import OpenAIEmbedding
    from graphrag.query.llm.oai.typing import OpenaiApiType
    from graphrag.query.question_gen.local_gen import LocalQuestionGen
    from graphrag.query.structured_search.local_search.mixed_context import (
        LocalSearchMixedContext,
    )
    from graphrag.query.structured_search.local_search.search import LocalSearch
    from graphrag.vector_stores.lancedb import LanceDBVectorStore

    INPUT_DIR = "ragtest/output/20240812-034710/artifacts"
    LANCEDB_URI = 'lancedb'

    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"
    RELATIONSHIP_TABLE = "create_final_relationships"
    COVARIATE_TABLE = "create_final_covariates"
    TEXT_UNIT_TABLE = "create_final_text_units"
    COMMUNITY_LEVEL = 2

    # read nodes table to get community and degree data
    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

    # load description embeddings to an in-memory lancedb vectorstore
    # to connect to a remote db, specify url and port values.
    description_embedding_store = LanceDBVectorStore(
        collection_name="entity_description_embeddings",
    )
    description_embedding_store.connect(db_uri=LANCEDB_URI)
    entity_description_embeddings = store_entity_semantic_embeddings(
        entities=entities, vectorstore=description_embedding_store
    )

    print(f"Entity count: {len(entity_df)}")
    entity_df.head()

    relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
    relationships = read_indexer_relationships(relationship_df)

    print(f"Relationship count: {len(relationship_df)}")
    relationship_df.head(13)

    # covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")

    # claims = read_indexer_covariates(covariate_df)

    # print(f"Claim records: {len(claims)}")
    # covariates = {"claims": claims}

    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

    print(f"Report records: {len(report_df)}")
    report_df.head()

    text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
    text_units = read_indexer_text_units(text_unit_df)

    print(f"Text unit records: {len(text_unit_df)}")
    text_unit_df.head()

    api_key = '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'
    llm_model = 'gpt-4o-mini'
    embedding_model = 'text-embedding-3-small'

    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
        max_retries=20,
    )

    token_encoder = tiktoken.get_encoding("cl100k_base")

    text_embedder = OpenAIEmbedding(
        api_key=api_key,
        api_base=None,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=20,
    )

    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        # covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )



    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 4,
        "top_k_relationships": 4,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
        "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    }

    llm_params = {
        "max_tokens": 2_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
        "temperature": 0.0,
    }

    st.session_state.search_engine = LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )



# Run the async function



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        prompt = f'{prompt}?' if prompt[-1] != '?' else prompt
        stream = st.session_state.search_engine.search(prompt).response
        print(stream)
        response = st.write_stream(stream_output(stream))
        st.session_state.messages.append({"role": "assistant", "content": response})


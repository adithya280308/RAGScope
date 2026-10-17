#RAG Week-5 Project(ADVANCED RAG TECHNIQUES)
from llama_index.core import SimpleDirectoryReader,Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_window_retriever import (
    create_nodes,
    create_sentence_window_index,
    create_query_engine,
)
import automerging_retriever
from Evaluations import answer_rel, context_rel, groundedness
from trulens.apps.llamaindex import TruLlama
from trulens.core import TruSession
from trulens.dashboard import run_dashboard
from trulens.core import FeedbackMode
import os



documents=SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".txt"]
).load_data()
  
if __name__=="__main__":

    llm = Groq(
        model="openai/gpt-oss-120b",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0
    )



    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
        )

    Settings.llm=llm
    Settings.embed_model=embed_model

    query="Explain RAG in simple terms."
    test_embedding = Settings.embed_model.get_query_embedding(query)

    print("Embedding type:", type(test_embedding))
    print("Embedding length:", len(test_embedding))

    print("First 5 values:", test_embedding[:5])
    
    nodes,node_parser=create_nodes(documents)
    index=create_sentence_window_index(nodes)
    sentence_window_query_engine=create_query_engine(6,3,index)

    response=sentence_window_query_engine.query(query)
    print("----------Sentence Window Retrieved Answer----------")
    print(response)

    nodes,leaf_nodes=automerging_retriever.create_nodes_and_leafnodes(documents)
    index=automerging_retriever.create_index(nodes,leaf_nodes)
    automerging_query_engine=automerging_retriever.build_automerging_query_engine(index,6,2)

  

    session=TruSession()

    feedbacks=[answer_rel,context_rel,groundedness]

    sentence_recorder=TruLlama(
        sentence_window_query_engine,
        app_name="Sentence Window RAG",
        app_version="v1",
        feedbacks=[],
        feedback_mode=FeedbackMode.WITH_APP_THREAD
    )

    automerging_recorder=TruLlama(
        automerging_query_engine,
        app_name="Automerging RAG",
        app_version="v1",
        feedbacks=[],
        feedback_mode=FeedbackMode.WITH_APP_THREAD
    )

    eval_questions=[]

    with open("Rag_Eval.txt", "r", encoding="utf-8") as file:
        for line in file:
            question = line.strip()

            if question:  # Skip blank lines
                eval_questions.append(question)

    print("Number of evaluation questions:", len(eval_questions))

    for i, question in enumerate(eval_questions, start=1):
        print(f"Question {i}: {question!r}")

    for question in eval_questions:
        embedding = Settings.embed_model.get_query_embedding(question)

        print(
            "Question:", repr(question),
            "| Embedding length:", len(embedding),
            "| Contains None:", any(value is None for value in embedding)
        )

    print("-------Sentence Retriever Evaluation Answers-------")
    with sentence_recorder as recording:
        for question in eval_questions:
            response=sentence_window_query_engine.query(question) 
            print(response)

    print("-------Automerging Retriever Eval ANswers--------")
    with automerging_recorder as recording:
        for question in eval_questions:
            response=automerging_query_engine.query(question) 
            print(response)
    
    print("Starting TruLens Dashboard...")
    run_dashboard(session)


    



    
        


    
    
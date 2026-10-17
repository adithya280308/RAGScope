#SENTENCE WINDOW RETRIER

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)


#CREATING NODES(i.e SENTENCES SPLITTED FROM THE DOC)
def create_nodes(documents):
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=4,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )

    nodes = node_parser.get_nodes_from_documents(documents)
    return nodes, node_parser

#CREATING INDEX
def create_sentence_window_index(nodes):
    
    index = VectorStoreIndex(
     nodes,
     embed_model=Settings.embed_model
    )
    return index

#CREATING RERANKER,POSTPROCESSOR,QUERY ENGINE
def create_query_engine(similarity_top_k, top_n, index):
    postproc = MetadataReplacementPostProcessor(
        target_metadata_key="window"
    )

    rerank = SentenceTransformerRerank(
        top_n=top_n,
        model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    sentence_window_query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[rerank, postproc]
    )

    return sentence_window_query_engine

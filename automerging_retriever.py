#AUTOMERGING RETRIEVER
from llama_index.core.node_parser import HierarchicalNodeParser,get_leaf_nodes
from llama_index.core import StorageContext,VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings

def create_nodes_and_leafnodes(documents):
    node_parser=HierarchicalNodeParser.from_defaults(chunk_sizes=[1024,256,64])
    nodes=node_parser.get_nodes_from_documents(documents)

    leaf_nodes=get_leaf_nodes(nodes)

    return nodes,leaf_nodes

#Storing Nodes in Doc store of storage_context and leaf_nodes in Vector_Store and creating index for leaf
def create_index(nodes,leaf_nodes):
    storage_context=StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    index=VectorStoreIndex(leaf_nodes,storage_context=storage_context,embed_model=Settings.embed_model)
    return index

def build_automerging_query_engine(index,similarity_top_k,top_n):
    base_retriever=index.as_retriever(similarity_top_k=similarity_top_k)

    retriever=AutoMergingRetriever(base_retriever,index.storage_context,simple_ratio_thresh=0.3,verbose=True)
    rerank=SentenceTransformerRerank(
        top_n=top_n,
        model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    automerging_query_engine=RetrieverQueryEngine.from_args(
        retriever,node_postprocessors=[rerank]
    )

    return automerging_query_engine




import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import sys, os, time, pprint, uuid, datetime, subprocess, json
import numpy as np
from pymilvus import connections
from pymilvus import (
   FieldSchema, DataType, 
   CollectionSchema, Collection)
from pymilvus import MilvusClient
import time
import numpy as np
import torch
from torch.nn import functional as F

##########
# Functions for IMDB demo notebook.
# Data source: Stanford AI Lab https://ai.stanford.edu/~amaas/data/sentiment/
##########

# Output words instead of scores.
def sentiment_score_to_name(score: float):
    if score > 0:
        return "Positive"
    elif score <= 0:
        return "Negative"

# Split data into train, valid, test. 
def partition_dataset(df_input, new_columns, smoke_test=False):
    """Splits data, assuming original, input dataframe contains 50K rows.

    Args:
        df_input (pandas.DataFrame): input data frame
        smoke_test (boolean): if True, use smaller number of rows for testing
    
    Returns:
        df_train, df_val, df_test (pandas.DataFrame): train, valid, test splits.
    """

    # Shuffle data and split into train/val/test.
    df_shuffled = df_input.sample(frac=1, random_state=1).reset_index()
    df_shuffled.columns = new_columns

    df_train = df_shuffled.iloc[:35_000]
    df_val = df_shuffled.iloc[35_000:40_000]
    df_test = df_shuffled.iloc[40_000:]

    # Save train/val/test split data locally in separate files.
    df_train.to_csv("train.csv", index=False, encoding="utf-8")
    df_val.to_csv("val.csv", index=False, encoding="utf-8")
    df_test.to_csv("test.csv", index=False, encoding="utf-8")

    return df_shuffled, df_train, df_val, df_test

# Function for experimenting with chunk_size.
def imdb_chunk_text(encoder, batch_size, df, chunk_size, chunk_overlap):

    batch = df.head(batch_size).copy()
    print(f"chunk size: {chunk_size}")
    print(f"original shape: {batch.shape}")
    
    start_time = time.time()
    # 1. Change primary key type to string.
    batch["movie_index"] = batch["movie_index"].apply(lambda x: str(x))

    # 2. Split the documents into smaller chunks and add as new column to batch df.
    batch['chunk'] = batch['text'].apply(recursive_splitter_wrapper, 
                                         chunk_size=chunk_size, 
                                         chunk_overlap=chunk_overlap)
    # Explode the 'chunk' column to create new rows for each chunk.
    batch = batch.explode('chunk', ignore_index=True)
    print(f"new shape: {batch.shape}")

    # 3. Add embeddings as new column in df.
    review_embeddings = torch.tensor(encoder.encode(batch['chunk']))
    # Normalize embeddings to unit length.
    review_embeddings = F.normalize(review_embeddings, p=2, dim=1)
    # Quick check if embeddings are normalized.
    norms = np.linalg.norm(review_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5) == True

    # 4. Convert embeddings to list of `numpy.ndarray`, each containing `numpy.float32` numbers.
    converted_values = list(map(np.float32, review_embeddings))
    batch['vector'] = converted_values

    # 5. Reorder columns for conveneince, so index first, labels at end.
    new_order = ["movie_index", "text", "chunk", "vector", "label_int", "label"]
    batch = batch[new_order]

    end_time = time.time()
    print(f"Chunking + embedding time for {batch_size} docs: {end_time - start_time} sec")

    # Inspect the batch of data.
    display(batch.head())
    # assert len(batch.chunk[0]) <= MAX_SEQ_LENGTH-1
    # assert len(batch.vector[0]) == EMBEDDING_LENGTH
    print(f"type embeddings: {type(batch.vector)} of {type(batch.vector[0])}")
    print(f"of numbers: {type(batch.vector[0][0])}")

    # Chunking looks good, drop the original text column.
    batch.drop(columns=["text"], inplace=True)

    return batch

# Function for embedding a query.
def embed_query(encoder, query):

    # Embed the query using same embedding model used to create the Milvus collection.
    query_embeddings = torch.tensor(encoder.encode(query))
    # Normalize embeddings to unit length.
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    # Quick check if embeddings are normalized.
    norms = np.linalg.norm(query_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5) == True
    # Convert the embeddings to list of list of np.float32.
    query_embeddings = list(map(np.float32, query_embeddings))

    return query_embeddings


##########
# Functions for LangChain chunking and embedding.
##########
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def recursive_splitter_wrapper(text, chunk_size, chunk_overlap):

    # Default chunk overlap is 10% chunk_size.
    chunk_overlap = np.round(chunk_size * 0.10, 0)

    # Use langchain's convenient recursive chunking method.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks: List[str] = text_splitter.split_text(text)

    # Replace special characters with spaces.
    chunks = [text.replace("<br /><br />", " ") for text in chunks]
    return chunks


##########
# Functions to process Milvus Client API responses.
##########

def client_assemble_retrieved_context(retrieved_top_k, metadata_fields=[], num_shot_answers=3):
    """ 
    For each question, assemble the context and metadata from the retrieved_top_k chunks.
    retrieved_top_k: list of dicts
    """
    # Assemble the context as a stuffed string.
    distances = []
    context = []
    context_metadata = []
    i = 1
    for r in retrieved_top_k[0]:
        distances.append(r['distance'])
        if i <= num_shot_answers:
            if len(metadata_fields) > 0:
                metadata = {}
                for field in metadata_fields:
                    metadata[field] = r['entity'][field]
                context_metadata.append(metadata)
            context.append(r['entity']['chunk'])
        i += 1

    # Assemble formatted results in a zipped list.
    formatted_results = list(zip(distances, context, context_metadata))
    # Return all the things for convenience.
    return formatted_results, context, context_metadata


##########
# Functions to process Milvus Search API responses.
##########

# Parse out the answer and context metadata from Milvus Search response.
def assemble_answer_sources(answer, context_metadata):
    """Assemble the answer and grounding sources into a string"""
    grounded_answer = f"Answer: {answer}\n"
    grounded_answer += "Grounding sources and citations:\n"

    for metadata in context_metadata:
        try:
            grounded_answer += f"'h1': {metadata['h1']}, 'h2':{metadata['h2']}\n"
        except:
            pass
        try:
            grounded_answer += f"'source': {metadata['source']}"
        except:
            pass
        
    return grounded_answer

# Stuff answers into a context string and stuff metadata into a list of dicts.
def assemble_retrieved_context(retrieved_results, metadata_fields=[], num_shot_answers=3):
    
    # Assemble the context as a stuffed string.
    # Also save the context metadata to retrieve along with the answer.
    context = []
    context_metadata = []
    i = 1
    for r in retrieved_results[0]:
        if i <= num_shot_answers:
            if len(metadata_fields) > 0:
                metadata = {}
                for field in metadata_fields:
                    metadata[field] = getattr(r.entity, field, None)
                context_metadata.append(metadata)
            context.append(r.entity.text)
        i += 1

    return context, context_metadata

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# sys.path.append("..")  # Adds higher directory to python modules path.
ENDPOINT = "http://192.168.1.166:19530"

model_name = "WhereIsAI/UAE-Large-V1"
encoder = SentenceTransformer(model_name, device=DEVICE)
#print(type(encoder))
#print(encoder)

# Get the model parameters and save for later.
EMBEDDING_DIM = encoder.get_sentence_embedding_dimension()
MAX_SEQ_LENGTH_IN_TOKENS = encoder.get_max_seq_length() 
MAX_SEQ_LENGTH = MAX_SEQ_LENGTH_IN_TOKENS
HF_EOS_TOKEN_LENGTH = 1

# Inspect model parameters.
#print(f"model_name: {model_name}")
#print(f"EMBEDDING_DIM: {EMBEDDING_DIM}")
#print(f"MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")

# connections.connect(uri = ENDPOINT)

chunk = str(sys.argv[1])

COLLECTION_NAME = "MilvusDocs"

# Add custom HNSW search index to the collection.
# M = max number graph connections per layer. Large M = denser graph.
# Choice of M: 4~64, larger M for larger data and larger embedding lengths.
M = 16
# efConstruction = num_candidate_nearest_neighbors per layer. 
# Use Rule of thumb: int. 8~512, efConstruction = M * 2.
efConstruction = M * 2
# Create the search index for local Milvus server.
INDEX_PARAMS = dict({
    'M': M,               
    "efConstruction": efConstruction })
index_params = {
    "index_type": "HNSW", 
    "metric_type": "COSINE", 
    "params": INDEX_PARAMS
    }

# Use no-schema Milvus client uses flexible json key:value format.
# https://milvus.io/docs/using_milvusclient.md
mc = MilvusClient(
    uri=ENDPOINT)

# Check if collection already exists, if so drop it.
#has = has_collection(COLLECTION_NAME)
#if has:
#    drop_result = drop_collection(COLLECTION_NAME)
#    print(f"Successfully dropped collection: `{COLLECTION_NAME}`")

# Create the collection.
mc.create_collection(COLLECTION_NAME, 
                     EMBEDDING_DIM,
                     consistency_level="Eventually", 
                     auto_id=True,  
                     overwrite=True,
                     # skip setting params below, if using AUTOINDEX
                     params=index_params
                    )

#print(f"Successfully created collection: `{COLLECTION_NAME}`")
#print(mc.describe_collection(COLLECTION_NAME))

chunk_list = []

# Generate embeddings using encoder from HuggingFace.
embeddings = torch.tensor(encoder.encode([chunk]))
embeddings = np.array(embeddings / np.linalg.norm(embeddings)) #use numpy
converted_values = list(map(np.float32, embeddings))[0]
chunk_dict = {
  'vector': converted_values,
  'chunk': chunk,
  'source': 'Medium'
}
chunk_list.append(chunk_dict)
start_time = time.time()
insert_result = mc.insert(
    COLLECTION_NAME,
    data=chunk_list,
    progress_bar=True)
end_time = time.time()
# print(f"Milvus Client insert time for {len(chunk_list)} vectors: {end_time - start_time} seconds")

row = {}
row['uuid'] = str(uuid.uuid4())
row['systemtime'] = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
row['ts'] =  int( time.time() )
row['starttime'] =  int( start_time )
row['endtime'] =  int( end_time )
row['milvusinsertresult'] = str(insert_result)
row['milvuscollectioname'] = str("MilvusDocs")
row['milvusdocsource'] = str("Medium")
row['maxseqlength'] = str(MAX_SEQ_LENGTH)
row['embeddingdim'] = str(EMBEDDING_DIM)

json_string = json.dumps(row)
print(json_string)

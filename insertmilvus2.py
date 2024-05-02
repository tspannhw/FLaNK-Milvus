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
import urllib.parse

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# sys.path.append("..")  # Adds higher directory to python modules path.
ENDPOINT = "http://192.168.1.166:19530"

model_name = "WhereIsAI/UAE-Large-V1"
encoder = SentenceTransformer(model_name, device=DEVICE)

# Get the model parameters and save for later.
EMBEDDING_DIM = encoder.get_sentence_embedding_dimension()
MAX_SEQ_LENGTH_IN_TOKENS = encoder.get_max_seq_length() 
MAX_SEQ_LENGTH = MAX_SEQ_LENGTH_IN_TOKENS
HF_EOS_TOKEN_LENGTH = 1

chunk = str(sys.argv[1])
filename = ""
try:
   filename = str(sys.argv[2])
except:
   filename = ""

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
mc = MilvusClient(uri=ENDPOINT)

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
  'source': 'Medium',
  'filename': filename
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
row['milvusinsertresult'] = urllib.parse.quote_plus(str(insert_result))
row['milvuscollectioname'] = str("MilvusDocs")
row['milvusdocsource'] = str("Medium")
row['milvusfilename'] = str(filename)
row['maxseqlength'] = str(MAX_SEQ_LENGTH)
row['embeddingdim'] = str(EMBEDDING_DIM)

json_string = json.dumps(row)
print(json_string)

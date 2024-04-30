import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import urllib.parse
from transformers import AutoTokenizer, pipeline
import sys, os, time, pprint, uuid, datetime, subprocess, json
import numpy as np
import numpy as np
from pymilvus import MilvusClient
from pymilvus import connections
from pymilvus import (
   FieldSchema, DataType, 
   CollectionSchema, Collection)
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
sys.path.append("..")  # Adds higher directory to python modules path.
ENDPOINT = "http://192.168.1.166:19530"
model_name = "WhereIsAI/UAE-Large-V1"
encoder = SentenceTransformer(model_name, device=DEVICE)

# Get the model parameters and save for later.
EMBEDDING_DIM = encoder.get_sentence_embedding_dimension()
MAX_SEQ_LENGTH_IN_TOKENS = encoder.get_max_seq_length() 
MAX_SEQ_LENGTH = MAX_SEQ_LENGTH_IN_TOKENS
HF_EOS_TOKEN_LENGTH = 1

# Thanks: https://medium.com/@zilliz_learn/building-an-open-source-chatbot-using-langchain-and-milvus-in-under-5-minutes-224c4d60ed19

connections.connect(uri = ENDPOINT)
question = str(sys.argv[1])
query = [question]

fields = [
   FieldSchema("pk", DataType.INT64, is_primary=True, auto_id=True),
   FieldSchema("vector", DataType.FLOAT_VECTOR, dim=768),]

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

mc = MilvusClient(uri=ENDPOINT)
mc.create_collection(COLLECTION_NAME, 
                     EMBEDDING_DIM,
                     consistency_level="Eventually", 
                     auto_id=True,  
                     overwrite=True,
                     params=index_params
                    )

encoder = SentenceTransformer(model_name, device=DEVICE)
embedded_question = torch.tensor(encoder.encode([question]))
embedded_question = F.normalize(embedded_question, p=2, dim=1)
embedded_question = list(map(np.float32, embedded_question))

TOP_K = 5

start_time = time.time()
results = mc.search(
    collection_name = COLLECTION_NAME,
    data=embedded_question, 
    anns_field="vector", 
    output_fields=["text", "source", "chunk"], 
    limit=TOP_K,
    consistency_level="Eventually")

elapsed_time = time.time() - start_time
rownum = 0

for r in results[0]:
    row = {}
    rownum = rownum + 1
    row['uuid'] = str(uuid.uuid4())
    row['systemtime'] = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
    row['ts'] =  int( time.time() )
    row['starttime'] =  int( start_time )
    row['elapsedtime'] =  int( elapsed_time )
    row['rownum'] = int ( rownum )
    #urllib.parse.quote_plus(str( result ))
    row['milvuschunk'] = str(r['entity']['chunk'])
    row['milvuscollectioname'] = str(COLLECTION_NAME)
    row['milvusdocsource'] = str(r['entity']['source'])
    json_string = json.dumps(row)
    print(json_string)


import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import sys, os, time, pprint, uuid, datetime, subprocess, json
import numpy as np
from pymilvus import connections
from pymilvus import (
   FieldSchema, DataType,
   CollectionSchema, Collection)
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# sys.path.append("..")  # Adds higher directory to python modules path.
model_name = "BAAI/bge-base-en-v1.5"
ENDPOINT = "http://localhost:19530"

connections.connect(uri = ENDPOINT)

fields = [
   FieldSchema("pk", DataType.INT64, is_primary=True, auto_id=True),
   FieldSchema("vector", DataType.FLOAT_VECTOR, dim=768),]

schema = CollectionSchema(
   fields,
   enable_dynamic_field=True)

mc = Collection("MilvusDocs", schema)

mc.create_index(
   field_name="vector",
   index_params={
       "index_type": "AUTOINDEX",
       "metric_type": "COSINE"})

encoder = SentenceTransformer(model_name, device=DEVICE)

chunk = str(sys.argv[1])

# Get the model parameters and save for later.

MAX_SEQ_LENGTH = encoder.get_max_seq_length()

EMBEDDING_LENGTH = encoder.get_sentence_embedding_dimension()

embeddings = torch.tensor(encoder.encode([chunk]))

embeddings = F.normalize(embeddings, p=2, dim=1)

converted_values = list(map(np.float32, embeddings))[0]

chunk_list = []

chunk_dict = {
    'vector': converted_values,
    'text': chunk,
    'source': "medium"}

chunk_list.append(chunk_dict)
insert_result = mc.insert(chunk_list)
mc.flush()

row = {}
row['uuid'] = str(uuid.uuid4())
row['systemtime'] = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
row['ts'] =  int( time.time() )
row['milvusdbname'] = str(mc.partitions[0].name)
row['milvusinsertresult'] = str(insert_result)
row['milvuscollectioname'] = str("MilvusDocs")
row['milvusdocsource'] = str("Medium")
row['maxseqlength'] = str(MAX_SEQ_LENGTH)
row['embeddinglength'] = str(EMBEDDING_LENGTH)

json_string = json.dumps(row)
print(json_string)

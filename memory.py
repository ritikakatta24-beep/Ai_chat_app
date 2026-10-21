import chromadb

client = chromadb.PersistentClient(path="chroma_data")
col = client.get_or_create_collection("test_memories")

col.add(
    ids=["1", "2", "3"],
    documents=[
        "Ritika told Tae she skipped lunch because of a lab exam.",
        "Ritika's favourite drink is cold coffee.",
        "Ritika has a classmate named Rahul who she studies with.",
    ],
)

results = col.query(
    query_texts=["what does she like to drink?"],
    n_results=1,
)
print(results["documents"])
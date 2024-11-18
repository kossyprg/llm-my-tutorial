from langsmith import Client

DATASET_NAME = "走れメロスRAGデータセット"

examples = [
    ("メロスは太陽の沈む速度の何倍で走りましたか？", "メロスは沈んでゆく太陽の十倍も早く走りました。"),
    ("メロスの妹の年齢はいくつか？", "16歳")
]

def create_dataset_if_not_exist():
    client_langsmith = Client()
    if not client_langsmith.has_dataset(dataset_name=DATASET_NAME):
        print("Create dataset")
        dataset = client_langsmith.create_dataset(dataset_name=DATASET_NAME)
        inputs, outputs = zip(
            *[({"question": question}, {"answer": answer}) for question, answer in examples]
        )
        client_langsmith.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
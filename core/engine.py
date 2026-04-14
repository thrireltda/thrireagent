import os, json, requests, chromadb, time
from fastapi import BackgroundTasks

class Engine:
    def get_collection(self):
        memory_endpoint = os.getenv("MEMORY_ENDPOINT")
        memory_port = os.getenv("MEMORY_PORT")
        collection_name = os.getenv("COLLECTION_NAME")
        return chromadb.HttpClient(host=memory_endpoint,port=memory_port).get_or_create_collection(collection_name)
    def get_embedding(self, cortex_endpoint: str, text: str):
        return requests.post(f"{cortex_endpoint}/api/embeddings", json={"model": "nomic-embed-text", "prompt": text}).json().get("embedding")
    def load_state(self, profile_file_path : str) -> str:
        try:
            with open(profile_file_path, 'r') as f:
                content = f.read()
                return content if content.strip() else "{}"
        except Exception:
            return "{}"
    def save_state(self, raw_json: str, profile_file_path: str):
        try:
            json_data = json.loads(raw_json) 
            with open(profile_file_path, 'w') as f:
                json.dump(json_data, f, indent=4)
        except Exception as e:
            print(f"Exception: {e}")
    def process(self, profile_file_path: str, prompt: str, model_name: str, cortex_endpoint: str, background_tasks: BackgroundTasks) -> str:
        current_state = self.load_state(profile_file_path)
        results = self.get_collection().query(query_embeddings=[self.get_embedding(cortex_endpoint, prompt)], n_results=5)
        context = "\n".join(results.get('documents', [[]])[0])
        payload = {
            "model": model_name,
            "messages": [ {"role": "system", "content": f"{current_state}:{context}"}, {"role": "user", "content": f"{prompt}"} ],
            "stream": False
        }
        response = requests.post(f"{cortex_endpoint}/api/chat", json=payload).json()['message']['content']
        text = f"{prompt}\n{response}"
        ts = str(time.time())
        self.get_collection().add(ids=[ts], embeddings=[self.get_embedding(cortex_endpoint, text)], documents=[f"{text}"])
        background_tasks.add_task(self.sync_state, profile_file_path, model_name, cortex_endpoint, f"{text}")
        return prompt
    def sync_state(self, profile_file_path: str, model_name: str, cortex_endpoint: str, chunk: str):
        current_state = self.load_state(profile_file_path)
        payload = {
            "model": model_name,
            "messages": [{"role": "system", "content": f"{current_state}"}, {"role": "user", "content": f"{chunk}"}],
            "stream": False,
            "format": "json"
        }
        response = requests.post(f"{cortex_endpoint}/api/chat", json=payload).json()
        self.save_state(response['message']['content'], profile_file_path)
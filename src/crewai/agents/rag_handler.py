from typing import Dict
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration


class RAGModelHandler:
    def __init__(self):
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
        self.model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=self.retriever)

    def query_rag_model(self, query: str, file_contents: Dict[str, str]) -> str:
        combined_query = f"{query}\n\n"
        for path, content in file_contents.items():
            combined_query += f"Content from {path}:\n{content}\n\n"

        input_ids = self.tokenizer(combined_query, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

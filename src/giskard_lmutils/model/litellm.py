import os
from typing import Optional
from litellm import completion, acompletion, embedding, aembedding

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class _LocalEmbeddingModel:
    def __init__(self, model: str):
        if not TORCH_AVAILABLE:
                raise ValueError("""
                torch is not installed. Please install it with `pip install torch`.
                This is required to use the local embedding model.
                Alternatively, you can use the remote embedding model by setting `is_local=False` in the embedding_params.
                """)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ValueError("""
            transformers is not installed. Please install it with `pip install transformers`.
            This is required to use the local embedding model.
            Alternatively, you can use the remote embedding model by setting `is_local=False` in the embedding_params.
            """)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


    def get_embedding(self, input: str):
        inputs = self.tokenizer(
            input, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)
    
class LiteLLMModel:

    def __init__(self, 
                 completion_model: Optional[str] = None, 
                 embedding_model: Optional[str] = None,
                 completion_params: Optional[dict] = None,
                 embedding_params: Optional[dict] = None,
                 env_prefix: str = 'GSK'):
        completion_model = completion_model or os.getenv(f'{env_prefix}_COMPLETION_MODEL')
        embedding_model = embedding_model or os.getenv(f'{env_prefix}_EMBEDDING_MODEL')

        if completion_model is None and embedding_model is None:
            raise ValueError("Either completion_model or embedding_model must be provided")
        
        if embedding_params is not None and embedding_params.get('is_local', False):
            self._local_embedding_model = _LocalEmbeddingModel(embedding_model)

        self._completion_params = {**(completion_params or {}), 'model': completion_model }
        self._embedding_params = {**(embedding_params or {}), 'model': embedding_model }

    def _build_completion_params(self, completion_params, messages):
        return {**self._completion_params, **completion_params, 'messages': messages}

    def _build_embedding_params(self, embedding_params, input):
        return {**self._embedding_params, **embedding_params, 'input': input}

    def complete(self, messages: list, **completion_params):
        completion_params = self._build_completion_params(completion_params, messages)

        return completion(**completion_params)

    async def acomplete(self, messages: list, **completion_params):
        completion_params = self._build_completion_params(completion_params, messages)

        return await acompletion(**completion_params)

    
    def _local_embed(self, input: list[str]):
        return {"data": [
            { "embedding": torch.stack([self._local_embedding_model.get_embedding(d)]).flatten().tolist()} for d in input
            ]}

    def embed(self, input: list[str], **embedding_params):
        embedding_params = self._build_embedding_params(embedding_params, input)

        if embedding_params.get('is_local', False):
            return self._local_embed(input)

        return embedding(**embedding_params)
    
    async def aembed(self, input: list[str], **embedding_params):
        embedding_params = self._build_embedding_params(embedding_params, input)

        if embedding_params.get('is_local', False):
            return self._local_embed(input)

        return await aembedding(**embedding_params)

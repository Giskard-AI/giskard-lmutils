import os

class LiteLLMModel:

    def __init__(self, 
                 completion_model: str | None = None, 
                 embedding_model: str | None = None,
                 completion_params: dict | None = None,
                 embedding_params: dict | None = None,
                 env_prefix: str = 'GSK'):
        completion_model = completion_model or os.getenv(f'{env_prefix}_COMPLETION_MODEL')
        embedding_model = embedding_model or os.getenv(f'{env_prefix}_EMBEDDING_MODEL')

        if completion_model is None and embedding_model is None:
            raise ValueError("Either completion_model or embedding_model must be provided")

        self._completion_params = (completion_params or {}) | { 'model': completion_model }
        self._embedding_params = (embedding_params or {}) | { 'model': embedding_model }
        self.model = None
        self.tokenizer = None
        self.device = None

    def _build_completion_params(self, completion_params, messages):
        return self._completion_params | completion_params | {'messages': messages}

    def _build_embedding_params(self, embedding_params, input):
        return self._embedding_params | embedding_params | {'input': input}

    def complete(self, messages: list, **completion_params):
        from litellm import completion
        completion_params = self._build_completion_params(completion_params, messages)

        return completion(**completion_params)

    async def acomplete(self, messages: list, **completion_params):
        from litellm import acompletion
        completion_params = self._build_completion_params(completion_params, messages)

        return await acompletion(**completion_params)
    

    
    def _local_embed(self, input: list[str], **embedding_params):
        import torch
        from transformers import AutoTokenizer, AutoModel

        if self.model is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_params['model'])
            self.model = AutoModel.from_pretrained(embedding_params['model']).to(self.device)

        def _get_embedding(input: str):
            inputs = self.tokenizer(
                input, return_tensors="pt", truncation=True, padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze(0)

        embeddings = torch.stack([_get_embedding(d) for d in input])
        return {"data": [{"embedding": embeddings.tolist()}]}

    
    def embed(self, input: list[str], **embedding_params):
       
        from litellm import embedding
        embedding_params = self._build_embedding_params(embedding_params, input)

        if embedding_params.get('is_local', False):
            return self._local_embed(**embedding_params)

        return embedding(**embedding_params)
    
    async def aembed(self, input: list[str], **embedding_params):
        from litellm import aembedding
        embedding_params = self._build_embedding_params(embedding_params, input)

        if embedding_params.get('is_local', False):
            return self._local_embed(**embedding_params)

        return await aembedding(**embedding_params)

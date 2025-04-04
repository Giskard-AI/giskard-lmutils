import os

class LiteLLMModel:

    def __init__(self, 
                 completion_model: str | None = None, 
                 embedding_model: str | None = None,
                 litellm_params: dict | None = None,
                 embedding_params: dict | None = None,
                 env_prefix: str = 'GSK'):
        completion_model = completion_model or os.getenv(f'{env_prefix}_COMPLETION_MODEL')
        embedding_model = embedding_model or os.getenv(f'{env_prefix}_EMBEDDING_MODEL')

        if completion_model is None and embedding_model is None:
            raise ValueError("Either completion_model or embedding_model must be provided")

        self._litellm_params = (litellm_params or {}) | { 'model': completion_model }
        self._embedding_params = (embedding_params or {}) | { 'model': embedding_model }

    def _build_completion_params(self, completion_params, messages):
        return self._litellm_params | completion_params | {'messages': messages}

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
    
    def embed(self, input: list[str], **embedding_params):
        from litellm import embedding
        embedding_params = self._build_embedding_params(embedding_params, input)

        return embedding(**embedding_params)
    
    async def aembed(self, input: list[str], **embedding_params):
        from litellm import aembedding
        embedding_params = self._build_embedding_params(embedding_params, input)

        return await aembedding(**embedding_params)
        
        
        
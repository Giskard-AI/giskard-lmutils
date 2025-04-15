import os
from typing import Optional, Union

from litellm import (
    CustomStreamWrapper,
    EmbeddingResponse,
    ModelResponse,
    acompletion,
    aembedding,
    completion,
    embedding,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class _LocalEmbeddingModel:
    def __init__(self, model: str):
        if not TORCH_AVAILABLE:
            raise ValueError(
                """
                torch is not installed. Please install it with `pip install giskard-lmutils[local-embedding]`.
                This is required to use the local embedding model.
                Alternatively, you can use the remote embedding model by setting `is_local=False` in the embedding_params.
                """
            )

        if not TRANSFORMERS_AVAILABLE:
            raise ValueError(
                """
            transformers is not installed. Please install it with `pip install giskard-lmutils[local-embedding]`.
            This is required to use the local embedding model.
            Alternatively, you can use the remote embedding model by setting `is_local=False` in the embedding_params.
            """
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def get_embedding(self, input: str):
        inputs = self.tokenizer(
            input, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)


class LiteLLMModel:

    def __init__(
        self,
        completion_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        completion_params: Optional[dict] = None,
        embedding_params: Optional[dict] = None,
        env_prefix: str = "GSK",
    ):
        """
        Initialize the LiteLLMModel.

        Args:
            completion_model (Optional[str]): The model to use for completion, following the litellm format. If not provided, the environment variable GSK_COMPLETION_MODEL will be used.
            embedding_model (Optional[str]): The model to use for embedding, following the litellm format. If not provided, the environment variable GSK_EMBEDDING_MODEL will be used.
            completion_params (Optional[dict]): The additional parameters to use for completion. See litellm completion documentation for more details.
            embedding_params (Optional[dict]): The additional parameters to use for embedding. See litellm embedding documentation for more details. Additionally, if is_local is True, the model will be loaded locally. Be sure to install giskard-lmutils using `pip install giskard-lmutils[local-embedding]` to use this feature.
            env_prefix (str): The prefix to use for the environment variables. Defaults to 'GSK'.
        """
        completion_model = completion_model or os.getenv(
            f"{env_prefix}_COMPLETION_MODEL"
        )
        embedding_model = embedding_model or os.getenv(f"{env_prefix}_EMBEDDING_MODEL")

        if completion_model is None and embedding_model is None:
            raise ValueError(
                "Either completion_model or embedding_model must be provided"
            )

        if embedding_params is not None and embedding_params.get("is_local", False):
            self._local_embedding_model = _LocalEmbeddingModel(embedding_model)

        self._completion_params = {
            **(completion_params or {}),
            "model": completion_model,
        }
        self._embedding_params = {**(embedding_params or {}), "model": embedding_model}

    def _build_completion_params(self, completion_params, messages):
        return {**self._completion_params, **completion_params, "messages": messages}

    def _build_embedding_params(self, embedding_params, input):
        return {**self._embedding_params, **embedding_params, "input": input}

    def complete(
        self, messages: list, **completion_params
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        """
        Complete a message.

        Args:
            messages (list): The messages to complete.
            **completion_params (dict): The additional parameters to use for completion. See litellm completion documentation for more details. Those will be merged with the default parameters, overriding duplicates.
        Returns:
            ModelResponse: A response object containing the generated completion and associated metadata.
        """
        completion_params = self._build_completion_params(completion_params, messages)

        return completion(**completion_params)

    async def acomplete(
        self, messages: list, **completion_params
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        """
        Complete a message asynchronously.

        Args:
            messages (list): The messages to complete.
            **completion_params (dict): The additional parameters to use for completion. See litellm acompletion documentation for more details. Those will be merged with the default parameters, overriding duplicates.
        Returns:
        """
        completion_params = self._build_completion_params(completion_params, messages)

        return await acompletion(**completion_params)

    def _local_embed(self, input: list[str]) -> EmbeddingResponse:
        return EmbeddingResponse(
            data=[
                {
                    "embedding": torch.stack(
                        [self._local_embedding_model.get_embedding(d)]
                    )
                    .flatten()
                    .tolist()
                }
                for d in input
            ]
        )

    def embed(self, input: list[str], **embedding_params) -> EmbeddingResponse:
        """
        Embed a message.

        Args:
            input (list): The messages to embed.
            **embedding_params (dict): The additional parameters to use for embedding. See litellm embedding documentation for more details. Additionally, if is_local is True, the model will be loaded locally. Be sure to install giskard-lmutils using `pip install giskard-lmutils[local-embedding]` to use this feature.
        """
        embedding_params = self._build_embedding_params(embedding_params, input)

        if embedding_params.get("is_local", False):
            return self._local_embed(input)

        return embedding(**embedding_params)

    async def aembed(self, input: list[str], **embedding_params) -> EmbeddingResponse:
        """
        Embed a message asynchronously.

        Args:
            input (list): The messages to embed.
            **embedding_params (dict): The additional parameters to use for embedding. See litellm aembedding documentation for more details. Additionally, if is_local is True, the model will be loaded locally. Be sure to install giskard-lmutils using `pip install giskard-lmutils[local-embedding]` to use this feature.
        """
        embedding_params = self._build_embedding_params(embedding_params, input)

        if embedding_params.get("is_local", False):
            return self._local_embed(input)

        return await aembedding(**embedding_params)

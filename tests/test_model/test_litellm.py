from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm import Choices, EmbeddingResponse, ModelResponse

from giskard_lmutils import LiteLLMModel
from tests.data.embedding import EMBEDDING_VALUES

mock_completion_response = ModelResponse(
    choices=[Choices(message={"role": "assistant", "content": "This is an answer"})]
)
mock_completion = MagicMock(return_value=mock_completion_response)
mock_acompletion = AsyncMock(return_value=mock_completion_response)

mock_embedding_response = EmbeddingResponse(
    data=[{"embedding": EMBEDDING_VALUES["This is a test"]}]
)
mock_embedding = MagicMock(return_value=mock_embedding_response)
mock_aembedding = AsyncMock(return_value=mock_embedding_response)


def test_init_without_params():
    with pytest.raises(
        ValueError, match="Either completion_model or embedding_model must be provided"
    ):
        LiteLLMModel()


@patch("giskard_lmutils.model.litellm.completion", mock_completion)
def test_complete():
    litellm_model = LiteLLMModel("gpt-4")

    messages = [{"role": "user", "content": "This is a test"}]
    response = litellm_model.complete(messages)

    mock_completion.assert_called_with(model="gpt-4", messages=messages)
    assert response == mock_completion_response

    mock_completion.reset_mock()


@pytest.mark.asyncio
@patch("giskard_lmutils.model.litellm.acompletion", mock_acompletion)
async def test_acomplete():
    litellm_model = LiteLLMModel("gpt-4")

    messages = [{"role": "user", "content": "This is a test"}]
    response = await litellm_model.acomplete(messages)

    mock_acompletion.assert_called_with(model="gpt-4", messages=messages)
    assert response == mock_completion_response

    mock_acompletion.reset_mock()


@patch("giskard_lmutils.model.litellm.completion", mock_completion)
def test_complete_custom_params():
    litellm_model = LiteLLMModel("gpt-4", completion_params={"api_key": "api_key"})

    messages = [{"role": "user", "content": "This is a test"}]
    response = litellm_model.complete(messages, format="json")

    mock_completion.assert_called_with(
        model="gpt-4", messages=messages, api_key="api_key", format="json"
    )
    assert response == mock_completion_response

    mock_completion.reset_mock()


@pytest.mark.asyncio
@patch("giskard_lmutils.model.litellm.acompletion", mock_acompletion)
async def test_acomplete_custom_params():
    litellm_model = LiteLLMModel("gpt-4", completion_params={"api_key": "api_key"})

    messages = [{"role": "user", "content": "This is a test"}]
    response = await litellm_model.acomplete(messages, format="json")

    mock_acompletion.assert_called_with(
        model="gpt-4", messages=messages, api_key="api_key", format="json"
    )
    assert response == mock_completion_response

    mock_acompletion.reset_mock()


@patch("giskard_lmutils.model.litellm.embedding", mock_embedding)
def test_embedding():
    litellm_model = LiteLLMModel(embedding_model="text-embedding-ada-002")
    response = litellm_model.embed(["This is a test"])
    mock_embedding.assert_called_with(
        model="text-embedding-ada-002", input=["This is a test"]
    )
    assert response == mock_embedding_response

    mock_embedding.reset_mock()


@pytest.mark.asyncio
@patch("giskard_lmutils.model.litellm.aembedding", mock_aembedding)
async def test_aembedding():
    litellm_model = LiteLLMModel(embedding_model="text-embedding-ada-002")
    response = await litellm_model.aembed(["This is a test"])
    mock_aembedding.assert_called_with(
        model="text-embedding-ada-002", input=["This is a test"]
    )
    assert response == mock_embedding_response


@patch("giskard_lmutils.model.litellm.embedding", mock_embedding)
def test_embedding_custom_params():
    litellm_model = LiteLLMModel(
        embedding_model="text-embedding-ada-002",
        embedding_params={"api_key": "api_key"},
    )
    response = litellm_model.embed(["This is a test"], api_key="api_key")
    mock_embedding.assert_called_with(
        model="text-embedding-ada-002", input=["This is a test"], api_key="api_key"
    )
    assert response == mock_embedding_response


@pytest.mark.asyncio
@patch("giskard_lmutils.model.litellm.aembedding", mock_aembedding)
async def test_aembedding_custom_params():
    litellm_model = LiteLLMModel(
        embedding_model="text-embedding-ada-002",
        embedding_params={"api_key": "api_key"},
    )
    response = await litellm_model.aembed(["This is a test"], api_key="api_key")
    mock_aembedding.assert_called_with(
        model="text-embedding-ada-002", input=["This is a test"], api_key="api_key"
    )
    assert response == mock_embedding_response


def test_local_embedding():
    litellm_model = LiteLLMModel(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_params={"is_local": True},
    )
    response = litellm_model.embed(["This is a test"])
    assert response == mock_embedding_response

from giskard_lmutils import LiteLLMModel
from unittest.mock import patch, MagicMock, AsyncMock
from litellm import ModelResponse, Choices
import pytest

mock_response = ModelResponse(choices=[Choices(message={'role': 'assistant', 'content': 'This is an answer'})])
mock_completion = MagicMock(return_value=mock_response)
mock_acompletion = AsyncMock(return_value=mock_response)

@patch('litellm.completion', mock_completion)
def test_complete():
    litellm_model = LiteLLMModel("gpt-4")

    messages = [{'role': 'user', 'content': 'This is a test'}]
    response = litellm_model.complete(messages)

    mock_completion.assert_called_with(model="gpt-4", messages=messages)
    assert response == mock_response

    mock_completion.reset_mock()


@pytest.mark.asyncio
@patch('litellm.acompletion', mock_acompletion)
async def test_acomplete():
    litellm_model = LiteLLMModel("gpt-4")

    messages = [{'role': 'user', 'content': 'This is a test'}]
    response = await litellm_model.acomplete(messages)

    mock_acompletion.assert_called_with(model="gpt-4", messages=messages)
    assert response == mock_response

    mock_acompletion.reset_mock()



@patch('litellm.completion', mock_completion)
def test_complete_custom_params():
    litellm_model = LiteLLMModel("gpt-4", api_key='api_key')

    messages = [{'role': 'user', 'content': 'This is a test'}]
    response = litellm_model.complete(messages, format="json")

    mock_completion.assert_called_with(model="gpt-4", messages=messages, api_key='api_key', format="json")
    assert response == mock_response

    mock_completion.reset_mock()


@pytest.mark.asyncio
@patch('litellm.acompletion', mock_acompletion)
async def test_acomplete_custom_params():
    litellm_model = LiteLLMModel("gpt-4", api_key='api_key')

    messages = [{'role': 'user', 'content': 'This is a test'}]
    response = await litellm_model.acomplete(messages, format="json")

    mock_acompletion.assert_called_with(model="gpt-4", messages=messages, api_key='api_key', format="json")
    assert response == mock_response

    mock_acompletion.reset_mock()


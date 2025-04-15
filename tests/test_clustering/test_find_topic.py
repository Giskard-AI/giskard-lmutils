from unittest.mock import AsyncMock, patch

import pytest
from litellm import Choices, ModelResponse

from giskard_lmutils import LiteLLMModel
from giskard_lmutils.clustering import find_topic

mock_response = ModelResponse(
    choices=[Choices(message={"role": "assistant", "content": "some topic"})]
)
mock_acompletion = AsyncMock(return_value=mock_response)


@pytest.mark.asyncio
@patch("litellm.acompletion", mock_acompletion)
async def test_acomplete_custom_params():
    litellm_model = LiteLLMModel("gpt-4")

    topic = await find_topic(
        litellm_model,
        [
            "This is a document",
            "This is a document",
            "This is a document",
            "This is a document",
            "This is a document",
            "This is a document",
        ],
        "en",
    )

    mock_acompletion.assert_called_with(
        model="gpt-4",
        temperature=0.0,
        seed=1729,
        json_output=False,
        messages=[
            {
                "role": "system",
                "content": 'Your task is to define the topic which best represents a set of documents.\n\nYour are given below a list of documents and you must summarise ALL the documents as a topic.\n- The topic should be as meaningful as possible\n- The topic should be as concise as possible\n- The topic should be a sentence describing the content\n- Provide the topic in this language: en\n\nThe user will provide the documents, consisting in multiple paragraphs delimited by dashes "----------".\nYou must output a single sentence containing the topic, without any other wrapping text or markdown.\n',
            },
            {
                "role": "user",
                "content": "----------This is a document\n\n----------This is a document\n\n----------This is a document\n\n----------This is a document\n\n----------This is a document\n\n----------This is a document",
            },
        ],
    )
    assert topic == "some topic"
    mock_acompletion.reset_mock()

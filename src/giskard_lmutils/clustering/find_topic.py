import logging

import numpy as np

from ..model import LiteLLMModel

LOGGER = logging.getLogger(__name__)

TOPIC_SUMMARIZATION_PROMPT = """Your task is to define the topic which best represents a set of documents.

Your are given below a list of documents and you must summarise ALL the documents as a topic.
- The topic should be as meaningful as possible
- The topic should be as concise as possible
- The topic should be a sentence describing the content
- Provide the topic in this language: {language}

The user will provide the documents, consisting in multiple paragraphs delimited by dashes "----------".
You must output a single sentence containing the topic, without any other wrapping text or markdown.
"""


async def find_topic(
    model: LiteLLMModel,
    topic_documents: list[str],
    language: str,
    document_max_length: int = 500,  # TODO: move in config?
    topic_document_count: int = 10,  # TODO: move in config?
    seed: int = 1729,
) -> str:
    """
    Find a topic for a set of documents.

    Args:
        model (LiteLLMModel): The model to use for completion.
        topic_documents (list[str]): The documents to find a topic for.
        language (str): The language of the topic.
        document_max_length (int): The maximum length of a document. This is used to limit the size of the documents used for the topic.
        topic_document_count (int): The number of documents to use for the topic. This is used to limit the size of the documents used for the topic.
        seed (int): The seed for the random number generator.

    Returns:
        str: The topic.
    """
    LOGGER.debug("Create topic name from topic documents")
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(topic_documents)
    topics_str = "\n\n".join(
        [
            "----------" + doc[:document_max_length]
            for doc in topic_documents[:topic_document_count]
        ]
    )

    summary: str = (
        (
            await model.acomplete(
                [
                    {
                        "role": "system",
                        "content": TOPIC_SUMMARIZATION_PROMPT.format(language=language),
                    },
                    {"role": "user", "content": topics_str},
                ],
                temperature=0.0,
                seed=seed,
                json_output=False,
            )
        )
        .choices[0]
        .message.content
    )

    LOGGER.debug("Summary: %s", summary)
    return summary

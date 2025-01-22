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
        document_max_length: int = 500, # TODO: move in config?
        topic_document_count: int = 10, # TODO: move in config?
        seed: int = 1729
) -> str:
    LOGGER.debug("Create topic name from topic documents")

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(topic_documents)
    topics_str = "\n\n".join(
        [
            "----------" + doc[:document_max_length]
            for doc in topic_documents[:topic_document_count]
        ]
    )

    summary: str = (await model.acomplete(
        [{
            'role': "system",
            'content': TOPIC_SUMMARIZATION_PROMPT.format(language=language)
        }, {
            'role': "user",
            'content': topics_str
        }],
        temperature=0.0,
        seed=seed,
        json_output=False,
    )).choices[0].message.content

    LOGGER.debug("Summary: %s", summary)
    return summary
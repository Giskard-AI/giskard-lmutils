class LiteLLMModel:

    def __init__(self, model: str, **litellm_params):
        self._litellm_params = litellm_params | { 'model': model }

    def _build_completion_params(self, completion_params, messages):
        return self._litellm_params | completion_params | {'messages': messages}

    def complete(self, messages: list, **completion_params):
        from litellm import completion
        completion_params = self._build_completion_params(completion_params, messages)

        return completion(**completion_params)

    async def acomplete(self, messages: list, **completion_params):
        from litellm import acompletion
        completion_params = self._build_completion_params(completion_params, messages)

        return await acompletion(**completion_params)
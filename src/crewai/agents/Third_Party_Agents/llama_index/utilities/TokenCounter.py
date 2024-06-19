from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import tiktoken


def get_callback_manager(model_name: str = "gpt-4o"):
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(model_name).encode
    )
    return CallbackManager([token_counter])

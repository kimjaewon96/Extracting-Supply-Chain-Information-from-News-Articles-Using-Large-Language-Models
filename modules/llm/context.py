from abc import ABC, abstractmethod
from copy import deepcopy

from llm.model import LLMModel
from llm.template import compile_template, render_template
from pydantic import BaseModel
from transformers import PreTrainedTokenizer


class LLMContext(ABC):
    def __init__(self, template="", *, tokenizer: PreTrainedTokenizer):
        self.template = compile_template(template)
        self.tokenizer = tokenizer

    def set_template(self, template_str):
        self.template = compile_template(template_str)

    def _render(self, *args, **kwargs):
        return render_template(self.template, *args, **kwargs)

    @abstractmethod
    def render(self, input_text):
        input_len = self.tokenizer.decode(input_text)
        if input_len > 8196:
            raise Exception("Input Exceed Max Tokens")


class ChatDialogue(BaseModel):
    role: str
    content: str
    # time: str


class LLMChatContext(LLMContext):
    def __init__(self, tokenizer: PreTrainedTokenizer, template=None):
        super().__init__(
            template if template else tokenizer.chat_template, tokenizer=tokenizer
        )
        self._history: list[ChatDialogue] = [
            # ChatDialogue(role="system", content="You are an assistant.")
        ]

    def add_chat(self, role: str, content: str):
        self._history.append(ChatDialogue(role=role, content=content))

    def render(self, add_generation_prompt=True):
        history_copy = deepcopy(self._history)
        while len(history_copy) > 0:
            prompt = self._render(
                messages=history_copy,
                add_generation_prompt=add_generation_prompt,
                bos_token="",  # Tokenizer automatically adds bos
                eos=self.tokenizer.eos_token,
            )
            input_len = len(self.tokenizer.encode(prompt))
            if input_len > 4096:
                print("Critial error: exceeded length")
                raise ValueError("Critial error: exceeded length")
                history_copy.pop(0)
            else:
                break
        if len(history_copy) == 0:
            raise Exception("Input Exceeded Max Tokens")
        return prompt

    def generate(
        self,
        gen: LLMModel,
        prefill: str = "",
        generation_config: object = {},
    ):
        prompt = self.render()
        prompt += prefill
        output = gen.generate(prompt, generation_config)
        self._history.append(ChatDialogue(role="assistant", content=prefill + output))
        return output

    def reply(self, prompt: str, prefill: str, gen, generation_config: object = {}):
        self.add_chat(role="user", content=prompt)
        return self.generate(
            role="assistant",
            prefill=prefill,
            gen=gen,
            generation_config=generation_config,
        )

    def choice(self, prefill: str, choice: list[str], gen: LLMModel):
        prompt = self.render()
        prompt += prefill
        output = gen.choice(prompt, choice)
        self._history.append(ChatDialogue(role="assistant", content=prefill + output))
        return output

    def clear(self):
        self._history = []

    def __repr__(self):
        return self._render(messages=self._history)

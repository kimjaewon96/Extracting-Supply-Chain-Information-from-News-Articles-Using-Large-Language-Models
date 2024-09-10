from functools import lru_cache

from jinja2 import Template
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

jinja_env = ImmutableSandboxedEnvironment(
    trim_blocks=True,
    lstrip_blocks=True,
)


def raise_exception(message):
    raise TemplateError(message)


jinja_env.globals["raise_exception"] = raise_exception


@lru_cache
def compile_template(input: str) -> Template:
    return jinja_env.from_string(input)


def render_template(template: Template, *args, **kwargs) -> str:
    return template.render(*args, **kwargs)

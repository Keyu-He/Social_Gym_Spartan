import logging
import asyncio
import os
import re
import json
import time
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from litellm import acompletion
from litellm.exceptions import BadRequestError
from litellm.utils import supports_response_schema
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,
)
from typing import Any, cast, Literal, overload

import gin

from pydantic import validate_call
from rich import print
from rich.logging import RichHandler

from sotopia.database import EnvironmentProfile, RelationshipProfile
from sotopia.messages import ActionType, AgentAction, ScriptBackground
from sotopia.messages.message_classes import (
    ScriptInteraction,
    ScriptInteractionReturnType,
)
from sotopia.utils import format_docstring


from sotopia.generation_utils.output_parsers import (
    OutputParser,
    PydanticOutputParser,
    StrOutputParser,
    OutputType,
    EnvResponse,
    ScriptOutputParser,
)

# Configure logger
log = logging.getLogger("sotopia.generation")
log.setLevel(logging.INFO)

# Create console handler with rich formatting
console_handler = RichHandler(rich_tracebacks=True)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)


def fill_template(template: str, **kwargs: str) -> str:
    """Fill template with kwargs, ignoring missing keys."""
    for k, v in kwargs.items():
        template = template.replace(f"{{{k}}}", v)
    return template


# Add handler to logger
log.addHandler(console_handler)

# subject to future OpenAI changes
DEFAULT_BAD_OUTPUT_PROCESS_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
_TEMPERATURE_SENTINEL = object()

# Cache temperature support per (model, base_url) to avoid repeated bad requests
_TEMPERATURE_SUPPORT_CACHE: dict[tuple[str, str | None], bool] = {}

# Reasoning text from the most recent completion in this asyncio task. Thinking
# models served with a vLLM reasoning parser return it separately from the JSON
# action; callers that log responses read it right after awaiting. A ContextVar
# keeps concurrent episodes from seeing each other's reasoning.
LAST_REASONING: ContextVar[str | None] = ContextVar("last_reasoning", default=None)
GENERATION_OPTIONS: ContextVar[dict[str, Any]] = ContextVar(
    "generation_options", default={}
)
LAST_COMPLETION_METADATA: ContextVar[dict[str, Any] | None] = ContextVar(
    "last_completion_metadata", default=None
)
# Identity of the episode the current task is playing (set by the tournament runner) so
# per-call progress events can be attributed to it.
CURRENT_EPISODE: ContextVar[str] = ContextVar("current_episode", default="")


def log_progress_event(kind: str, **fields: Any) -> None:
    """Append one JSON line to SOTOPIA_PROGRESS_LOG (no-op when unset).

    kinds: "start" (an episode begins), "call" (one model response: ok / failed, with
    retry=True for a transient failure that will be retried), "end" (episode written).
    Read by experiments/run_progress.py to draw a live picture of a run.
    """
    path = os.environ.get("SOTOPIA_PROGRESS_LOG")
    if not path:
        return
    rec = {"ts": time.time(), "kind": kind, "episode": CURRENT_EPISODE.get(), **fields}
    try:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass
_GENERATION_OPTION_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "seed",
    "extra_body",
}


def _generation_options() -> dict[str, Any]:
    """Only inference controls may override the request, never its routing/schema."""
    options = deepcopy(GENERATION_OPTIONS.get())
    unknown = options.keys() - _GENERATION_OPTION_KEYS
    if unknown:
        raise ValueError(f"Unsupported generation options: {sorted(unknown)}")
    if "extra_body" in options:
        body = options["extra_body"]
        allowed = (_GENERATION_OPTION_KEYS - {"extra_body"}) | {"chat_template_kwargs"}
        if not isinstance(body, dict) or body.keys() - allowed:
            raise ValueError(
                "extra_body may contain only inference controls and chat_template_kwargs"
            )
        if "chat_template_kwargs" in body and not isinstance(
            body["chat_template_kwargs"], dict
        ):
            raise ValueError("chat_template_kwargs must be a dictionary")
        overlap = body.keys() & (options.keys() - {"extra_body"})
        if overlap:
            raise ValueError(
                f"Duplicate generation options in extra_body: {sorted(overlap)}"
            )
    return options


# Failed model calls, one rule (experiments/utils.turn_request_failed reads logs by the same rule):
# - A request failure is an exception from the completion call, a response with no choices, or a generation
#   the server aborted (finish_reason 'abort', vLLM's server-side abort, or 'error'; also no finish reason
#   and no content, where nothing shows a completed generation). It is not the model's doing: it is retried
#   with backoff when transient, and if it still fails the turn's log entry gets request_failed=True, so the
#   rating code drops the episode and the tournament runner replays it. A deterministic rejection (HTTP 400,
#   including a prompt over the context window) also gets permanent_failure=True: the runner reports it and
#   does not replay it, and it is still never rated.
# - Anything the server completed is the model's move, played as 'none' when it cannot be used, never retried
#   or repaired, and rated: JSON that does not parse, an empty reply that stopped normally, a reply cut off at
#   the token limit, a structured-output refusal (its text is kept in the log) or a reply a content filter
#   withheld. The log keeps the model's raw reply next to the error.
DEFAULT_ENDPOINT_RETRIES = 6
DEFAULT_ENDPOINT_RETRY_WAIT = 5.0  # seconds before the first retry; doubles each retry
DEFAULT_ENDPOINT_RETRY_MAX_WAIT = 60.0
_TRANSIENT_STATUS = {408, 409, 425, 429}
# finish_reason values that mean the server stopped the generation itself rather than completing it.
SERVER_ABORT_FINISH_REASONS = frozenset({"abort", "error"})


class EmptyCompletionError(RuntimeError):
    """The API answered without usable content (see _content_or_raise for which cases are the model's move)."""

    def __init__(self, message: str, finish_reason: Any = None) -> None:
        super().__init__(message)
        self.finish_reason = finish_reason


class ModelRefusalError(RuntimeError):
    """A structured-output refusal: the server completed the request and the model declined to answer."""

    def __init__(self, message: str, refusal: str, finish_reason: Any = None) -> None:
        super().__init__(message)
        self.refusal = refusal
        self.finish_reason = finish_reason


class StructuredOutputUnsupported(RuntimeError):
    """The model does not support a response schema (checked before the request is sent)."""


def _attach_reply(exc: BaseException, prompt: list[dict[str, str]], reply: str) -> None:
    """Keep the prompt and the model's raw reply on a model-output error, so the log shows the reply."""
    try:
        setattr(exc, "_sotopia_prompt", prompt)
        setattr(exc, "_sotopia_raw_reply", reply)
    except Exception:
        pass


def _mark_request_error(exc: BaseException) -> None:
    """The completion request itself failed (raised by acompletion)."""
    try:
        setattr(exc, "_sotopia_request_error", True)
    except Exception:
        pass


def _mark_model_output_error(exc: BaseException) -> None:
    """The request succeeded and the model's reply could not be used: the model's own move."""
    try:
        setattr(exc, "_sotopia_model_output", True)
    except Exception:
        pass


def is_model_output_error(exc: BaseException) -> bool:
    return bool(getattr(exc, "_sotopia_model_output", False))


def is_retryable_request_error(exc: BaseException) -> bool:
    """A failed request worth waiting out: connection or timeout errors, server errors, rate limits, a
    self-hosted server that is restarting (HTTP 404 or a model-loading message), a response with no choices
    or a generation the server aborted. A model's own reply is never retried, whatever text it contains. A
    request rejected with another 4xx status is retried only for a model-loading message: the generic words
    and status numbers in the list below are not looked for in its text ("max_tokens must be at most 500")."""
    if is_model_output_error(exc):
        return False
    if isinstance(exc, EmptyCompletionError):
        return True
    if not getattr(exc, "_sotopia_request_error", False):
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status >= 500 or status in _TRANSIENT_STATUS or status == 404):
        return True
    if isinstance(status, int) and 400 <= status < 500:
        return _model_loading_error(exc)
    if isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)):
        return True
    name = type(exc).__name__
    if name in {
        "APIConnectionError",
        "APITimeoutError",
        "Timeout",
        "InternalServerError",
        "ServiceUnavailableError",
        "BadGatewayError",
        "RateLimitError",
    }:
        return True
    return _transient_endpoint_error(exc)


def is_permanent_request_error(exc: BaseException) -> bool:
    """A request the server rejected and would reject again unchanged: HTTP 400 (a bad request, including a
    prompt longer than the model's context window) that is not a transient model-loading message. The turn
    is flagged as a failed request and also as permanent: the tournament runner reports it once and does
    not replay the roster on every resume; the episode is still never rated."""
    if is_model_output_error(exc) or not getattr(exc, "_sotopia_request_error", False):
        return False
    if is_retryable_request_error(exc):
        return False
    name = type(exc).__name__
    return name in {"ContextWindowExceededError", "BadRequestError"} or getattr(exc, "status_code", None) == 400


_SCHEMA_WORDS = ("response_schema", "response_format", "json_schema")


def is_schema_rejection(exc: BaseException) -> bool:
    """The structured-output request itself was refused: the capability check before the request, or an API
    error (HTTP 400/422) whose message is about the response schema. This is the only case in which
    agenerate_action plays the turn again without a schema. A model's reply never qualifies, whatever it
    says: its errors are model output, not request errors."""
    if is_model_output_error(exc):
        return False
    if isinstance(exc, StructuredOutputUnsupported):
        return True
    if not getattr(exc, "_sotopia_request_error", False):
        return False
    status = getattr(exc, "status_code", None)
    if status not in (400, 422) and type(exc).__name__ not in {"BadRequestError", "UnsupportedParamsError"}:
        return False
    message = str(exc).lower()
    return any(word in message for word in _SCHEMA_WORDS)


def _env_number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _retry_wait(retry_index: int) -> float:
    """Exponential backoff: SOTOPIA_ENDPOINT_RETRY_WAIT, doubled each retry, capped at
    SOTOPIA_ENDPOINT_RETRY_MAX_WAIT."""
    base = _env_number("SOTOPIA_ENDPOINT_RETRY_WAIT", DEFAULT_ENDPOINT_RETRY_WAIT)
    cap = _env_number("SOTOPIA_ENDPOINT_RETRY_MAX_WAIT", DEFAULT_ENDPOINT_RETRY_MAX_WAIT)
    return float(min(cap, base * (2**retry_index)))


def _reasoning_of(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    return getattr(message, "reasoning_content", None) or getattr(
        message, "reasoning", None
    )


@dataclass(frozen=True)
class TemperatureSetting:
    value: float | None
    treat_as_default: bool = False


def default_temperature(value: float | None) -> TemperatureSetting:
    return TemperatureSetting(value=value, treat_as_default=True)


def custom_temperature(value: float | None) -> TemperatureSetting:
    return TemperatureSetting(value=value, treat_as_default=False)


@validate_call
async def format_bad_output(
    ill_formed_output: str,
    format_instructions: str,
    model_name: str,
    use_fixed_model_version: bool = True,
    base_url: str | None = None,
) -> str:
    template = """
    Given the string that can not be parsed by json parser, reformat it to a string that can be parsed by json parser.
    Original string: {ill_formed_output}

    Format instructions: {format_instructions}

    Please only generate the JSON:
    """

    input_values = {
        "ill_formed_output": ill_formed_output,
        "format_instructions": format_instructions,
    }
    content = template.format(**input_values)

    # Build completion kwargs
    completion_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
    }

    # Parse format_instructions to get the schema
    try:
        schema = json.loads(format_instructions)

        def _fix_schema(s: dict[str, Any]) -> None:
            if s.get("type") == "array":
                if "prefixItems" in s:
                    # OpenAI doesn't support prefixItems (tuple validation).
                    # Convert to items: {anyOf: [...]} to satisfy "items must be a schema object"
                    # This allows valid tuple elements but loses positional validation, which is acceptable for strict=False.
                    prefix_items = s.pop("prefixItems")
                    s["items"] = {"anyOf": prefix_items}

                if "items" in s and isinstance(s["items"], dict):
                    _fix_schema(s["items"])
                elif "items" in s and isinstance(s["items"], list):
                    # Should not happen after the fix above, but handle legacy cases if any
                    for item in s["items"]:
                        _fix_schema(item)
            elif s.get("type") == "object":
                if "properties" in s:
                    for prop in s["properties"].values():
                        _fix_schema(prop)
                if "additionalProperties" in s and isinstance(
                    s["additionalProperties"], dict
                ):
                    _fix_schema(s["additionalProperties"])
                if "$defs" in s:
                    for def_schema in s["$defs"].values():
                        _fix_schema(def_schema)

        _fix_schema(schema)

        # Build proper json_schema response_format
        completion_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "reformatted_output",
                "schema": schema,
                "strict": False,
            },
        }
    except json.JSONDecodeError:
        # Fallback to json_object if schema parsing fails
        completion_kwargs["response_format"] = {"type": "json_object"}

    # Add base_url if provided
    if base_url is not None:
        completion_kwargs["base_url"] = base_url

    try:
        response = await acompletion(**completion_kwargs)
    except Exception as exc:
        _mark_request_error(exc)
        raise
    reformatted_output = response.choices[0].message.content
    assert isinstance(reformatted_output, str)
    log.debug(f"Model: {model_name}")
    log.debug(f"Prompt: {content}")
    log.info(f"Reformated output: {reformatted_output}")
    return reformatted_output


@overload
async def agenerate(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: OutputParser[OutputType],
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    structured_output: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: Literal[False] = False,
    repair_bad_output: bool = True,
) -> OutputType: ...


@overload
async def agenerate(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: OutputParser[OutputType],
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    structured_output: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: Literal[True] = ...,
    repair_bad_output: bool = True,
) -> tuple[OutputType, list[dict[str, str]], str]: ...


@gin.configurable
@validate_call
async def agenerate(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: OutputParser[OutputType],
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    structured_output: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: bool = False,
    repair_bad_output: bool = True,
) -> OutputType | tuple[OutputType, list[dict[str, str]], str]:
    """Generate text using LiteLLM instead of Langchain.

    repair_bad_output: in the unstructured path, a reply that does not parse is sent to a second model call
    (format_bad_output) to be reformatted. Game actions pass False (agenerate_action): a model's malformed
    reply is its own move, so it is raised as model output with the raw reply attached, never repaired."""
    # Format template with input values
    if "format_instructions" not in input_values:
        input_values["format_instructions"] = output_parser.get_format_instructions()

    # Process template
    template = format_docstring(template)

    # Replace template variables
    for key, value in input_values.items():
        template = template.replace(f"{{{key}}}", str(value))

    if model_name.startswith("custom"):
        base_url, api_key = (
            model_name.split("@")[1],
            os.environ.get("CUSTOM_API_KEY", "EMPTY"),
        )
        model_name = model_name.split("@")[0].replace("custom/", "openai/")
    else:
        base_url = None
        api_key = None

    cache_key = (model_name, base_url)
    generation_options = _generation_options()
    # Explicit per-agent settings take precedence over the helper's defaults.
    if "temperature" in generation_options:
        temperature = custom_temperature(generation_options["temperature"])

    supported_params: list[str] | None = None
    if base_url is None:
        supported_params = get_supported_openai_params(model=model_name)

    effective_temperature: float | None
    treat_as_default: bool
    user_provided_temperature: bool

    if temperature is _TEMPERATURE_SENTINEL:
        effective_temperature = DEFAULT_TEMPERATURE
        treat_as_default = True
        user_provided_temperature = False
    elif isinstance(temperature, TemperatureSetting):
        effective_temperature = temperature.value
        treat_as_default = temperature.treat_as_default
        user_provided_temperature = True
    else:
        effective_temperature = cast(float | None, temperature)
        treat_as_default = False
        user_provided_temperature = True

    send_temperature = (
        user_provided_temperature
        and effective_temperature is not None
        and not treat_as_default
    )
    non_default_requested = send_temperature and not treat_as_default

    if send_temperature:
        if cache_key in _TEMPERATURE_SUPPORT_CACHE:
            if not _TEMPERATURE_SUPPORT_CACHE[cache_key]:
                send_temperature = False
                if non_default_requested:
                    log.warning(
                        "Model %s previously rejected temperature; ignoring temperature=%s",
                        model_name,
                        effective_temperature,
                    )
        elif supported_params is not None and "temperature" not in supported_params:
            _TEMPERATURE_SUPPORT_CACHE[cache_key] = False
            send_temperature = False
            if non_default_requested:
                log.warning(
                    "Model %s does not support temperature; ignoring temperature=%s",
                    model_name,
                    effective_temperature,
                )

    async def _call_with_retry(completion_kwargs: dict[str, Any]) -> Any:
        nonlocal send_temperature
        call_kwargs = dict(completion_kwargs)
        call_kwargs.update(generation_options)
        call_kwargs.pop("temperature", None)
        if send_temperature:
            call_kwargs["temperature"] = effective_temperature
        attempts: list[dict[str, Any]] = []

        async def complete() -> Any:
            # These are the kwargs sent to LiteLLM, not inferred server defaults.
            options = deepcopy(
                {k: v for k, v in call_kwargs.items() if k in _GENERATION_OPTION_KEYS}
            )
            attempt: dict[str, Any] = {
                "effective_options": options,
                "structured_output": "response_format" in call_kwargs,
            }
            started = time.perf_counter()
            finish_reason = None
            usage = None
            error = None
            try:
                response = await acompletion(**call_kwargs)
                choices = getattr(response, "choices", None)
                if choices:
                    finish_reason = getattr(choices[0], "finish_reason", None)
                usage = getattr(response, "usage", None)
                if hasattr(usage, "model_dump"):
                    usage = usage.model_dump()
                elif hasattr(usage, "dict"):
                    usage = usage.dict()
                return response
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
                _mark_request_error(exc)
                raise
            finally:
                attempt.update(
                    elapsed_seconds=time.perf_counter() - started,
                    finish_reason=finish_reason,
                    usage=deepcopy(usage),
                )
                if error:
                    attempt["error"] = error
                attempts.append(attempt)
                LAST_COMPLETION_METADATA.set(
                    {
                        "finish_reason": finish_reason,
                        "usage": deepcopy(usage),
                        "elapsed_seconds": sum(a["elapsed_seconds"] for a in attempts),
                        "effective_options": options,
                        "attempts": deepcopy(attempts),
                        **({"error": error} if error else {}),
                    }
                )

        try:
            response = await complete()
            if send_temperature:
                _TEMPERATURE_SUPPORT_CACHE[cache_key] = True
            return response
        except BadRequestError as exc:
            if send_temperature and "temperature" in str(exc).lower():
                send_temperature = False
                _TEMPERATURE_SUPPORT_CACHE[cache_key] = False
                if non_default_requested:
                    log.warning(
                        "Model %s does not support temperature; ignoring temperature=%s",
                        model_name,
                        effective_temperature,
                    )
                call_kwargs.pop("temperature", None)
                return await complete()
            raise

    if structured_output:
        if not base_url:
            assert supported_params is not None
            # Checked before the request is sent; agenerate_action then plays the turn without a schema.
            if "response_format" not in supported_params:
                raise StructuredOutputUnsupported("response_format is not supported in this model")
            if not supports_response_schema(model=model_name):
                raise StructuredOutputUnsupported("response_schema is not supported in this model")
        messages = [{"role": "user", "content": template}]

        assert isinstance(
            output_parser, PydanticOutputParser
        ), "structured output only supported in PydanticOutputParser"
        completion_kwargs = dict(
            model=model_name,
            messages=messages,
            response_format=output_parser.pydantic_object,
            drop_params=True,  # drop params to avoid model error if the model does not support it
            base_url=base_url,
            api_key=api_key,
        )
        response = await _call_with_retry(completion_kwargs)
        LAST_REASONING.set(_reasoning_of(response))
        try:
            result = _content_or_raise(response, model_name)
        except Exception as exc:
            if is_model_output_error(exc):
                _attach_reply(exc, messages, getattr(exc, "refusal", ""))
            raise
        result, inline_reasoning = _split_inline_think(result)
        LAST_REASONING.set(_reasoning_of(response) or inline_reasoning)
        # Include agent name in logs if available
        agent_name = input_values.get("agent", "")
        log_prefix = f" [{agent_name}]" if agent_name else ""
        log.debug(f"Model: {model_name}")
        log.debug(f"Prompt: {messages}")
        try:
            clean_result = json.dumps(json.loads(result.strip()), ensure_ascii=False)
        except Exception:
            clean_result = result.replace("\n", "").strip()
            clean_result = re.sub(r"\s+", " ", clean_result)
        log.info(f"Generated result{log_prefix}: {clean_result}")
        assert isinstance(result, str)
        try:
            parsed = cast(OutputType, output_parser.parse(result))
        except Exception as exc:
            _mark_model_output_error(exc)
            _attach_reply(exc, messages, result)
            raise
        if return_prompt_and_response:
            return parsed, messages, result
        return parsed

    messages = [{"role": "user", "content": template}]

    completion_kwargs = dict(
        model=model_name,
        messages=messages,
        drop_params=True,
        base_url=base_url,
        api_key=api_key,
    )
    response = await _call_with_retry(completion_kwargs)
    LAST_REASONING.set(_reasoning_of(response))
    try:
        result = _content_or_raise(response, model_name)
    except Exception as exc:
        if is_model_output_error(exc):
            _attach_reply(exc, messages, getattr(exc, "refusal", ""))
        raise

    try:
        parsed_result = output_parser.parse(result)
    except Exception as e:
        if isinstance(output_parser, ScriptOutputParser) or not repair_bad_output:
            # The model's own move: never sent to a second call to be repaired.
            _mark_model_output_error(e)
            _attach_reply(e, messages, result)
            raise e
        log.debug(
            f"Failed to parse result: {result}\nEncounter Exception {e}\nstart to reparse",
            extra={"markup": True},
        )
        # Handle bad output reformatting
        reformat_result = await format_bad_output(
            result,
            output_parser.get_format_instructions(),
            bad_output_process_model or model_name,
            use_fixed_model_version,
            base_url=base_url,
        )
        try:
            parsed_result = output_parser.parse(reformat_result)
        except Exception as exc:
            _mark_model_output_error(exc)
            raise

    # Include agent name in logs if available
    agent_name = input_values.get("agent", "")
    log_prefix = f" [{agent_name}]" if agent_name else ""
    log.debug(f"Model: {model_name}")
    log.debug(f"Prompt: {messages}")
    try:
        clean_result = json.dumps(json.loads(result.strip()), ensure_ascii=False)
    except Exception:
        clean_result = result.replace("\n", "").strip()
        clean_result = re.sub(r"\s+", " ", clean_result)
    log.info(f"Generated result{log_prefix}: {clean_result}")
    if return_prompt_and_response:
        return parsed_result, messages, result
    return parsed_result


@overload
async def agenerate_env_profile(
    model_name: str,
    inspiration_prompt: str = "asking my boyfriend to stop being friends with his ex",
    examples: str = "",
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: Literal[False] = False,
) -> EnvironmentProfile: ...


@overload
async def agenerate_env_profile(
    model_name: str,
    inspiration_prompt: str = "asking my boyfriend to stop being friends with his ex",
    examples: str = "",
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: Literal[True] = ...,
) -> tuple[EnvironmentProfile, list[dict[str, str]], str]: ...


@gin.configurable
@validate_call
async def agenerate_env_profile(
    model_name: str,
    inspiration_prompt: str = "asking my boyfriend to stop being friends with his ex",
    examples: str = "",
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: bool = False,
) -> EnvironmentProfile | tuple[EnvironmentProfile, list[dict[str, str]], str]:
    """
    Using langchain to generate the background
    """
    if return_prompt_and_response:
        return await agenerate(
            model_name=model_name,
            template="""Please generate scenarios and goals based on the examples below as well as the inspirational prompt, when creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it.
            Examples:
            {examples}
            Inspirational prompt: {inspiration_prompt}
            Please use the following format:
            {format_instructions}
            """,
            input_values=dict(
                inspiration_prompt=inspiration_prompt,
                examples=examples,
            ),
            output_parser=PydanticOutputParser(pydantic_object=EnvironmentProfile),
            temperature=temperature,
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
            return_prompt_and_response=True,
        )
    else:
        return await agenerate(
            model_name=model_name,
            template="""Please generate scenarios and goals based on the examples below as well as the inspirational prompt, when creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it.
            Examples:
            {examples}
            Inspirational prompt: {inspiration_prompt}
            Please use the following format:
            {format_instructions}
            """,
            input_values=dict(
                inspiration_prompt=inspiration_prompt,
                examples=examples,
            ),
            output_parser=PydanticOutputParser(pydantic_object=EnvironmentProfile),
            temperature=temperature,
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
            return_prompt_and_response=False,
        )


@overload
async def agenerate_relationship_profile(
    model_name: str,
    agents_profiles: list[str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: Literal[False] = False,
) -> RelationshipProfile: ...


@overload
async def agenerate_relationship_profile(
    model_name: str,
    agents_profiles: list[str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: Literal[True] = ...,
) -> tuple[RelationshipProfile, list[dict[str, str]], str]: ...


@validate_call
async def agenerate_relationship_profile(
    model_name: str,
    agents_profiles: list[str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: bool = False,
) -> RelationshipProfile | tuple[RelationshipProfile, list[dict[str, str]], str]:
    """
    Using langchain to generate the background
    """
    agent_profile = "\n".join(agents_profiles)
    if return_prompt_and_response:
        return await agenerate(
            model_name=model_name,
            template="""Please generate relationship between two agents based on the agents' profiles below. Note that you generate
            {agent_profile}
            Please use the following format:
            {format_instructions}
            """,
            input_values=dict(
                agent_profile=agent_profile,
            ),
            output_parser=PydanticOutputParser(pydantic_object=RelationshipProfile),
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
            return_prompt_and_response=True,
        )
    else:
        return await agenerate(
            model_name=model_name,
            template="""Please generate relationship between two agents based on the agents' profiles below. Note that you generate
            {agent_profile}
            Please use the following format:
            {format_instructions}
            """,
            input_values=dict(
                agent_profile=agent_profile,
            ),
            output_parser=PydanticOutputParser(pydantic_object=RelationshipProfile),
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
            return_prompt_and_response=False,
        )


@overload
async def agenerate_action(
    model_name: str,
    history: str,
    turn_number: int,
    action_types: list[ActionType],
    agent: str,
    goal: str,
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    script_like: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    strict_action_constraint: bool = False,
    custom_template: str | None = None,
    return_prompt_and_response: Literal[False] = False,
    structured_output: bool = True,
) -> AgentAction: ...


@overload
async def agenerate_action(
    model_name: str,
    history: str,
    turn_number: int,
    action_types: list[ActionType],
    agent: str,
    goal: str,
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    script_like: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    strict_action_constraint: bool = False,
    custom_template: str | None = None,
    return_prompt_and_response: Literal[True] = ...,
    structured_output: bool = True,
) -> tuple[AgentAction, list[dict[str, str]], str]: ...



_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def _split_inline_think(text: str) -> tuple[str, str | None]:
    """Remove ``<think>...</think>`` blocks a model wrote inline and return them separately.

    Models served without a reasoning parser (or prompted to reason in a scratchpad before the
    JSON action) put their reasoning in the text itself; it is logged as the turn's reasoning
    and kept out of the action parser.
    """
    blocks = _THINK_RE.findall(text)
    if not blocks:
        return text, None
    return _THINK_RE.sub("", text).strip(), "\n\n".join(b.strip() for b in blocks)


def _content_or_raise(response: Any, model_name: str) -> str:
    """Return the completion text, or raise. The rule (see the comment above DEFAULT_ENDPOINT_RETRIES):

    - A response with no choices (or a choice without a message), a generation the server aborted
      (finish_reason 'abort' or 'error', even with partial text), or one with neither content nor a
      finish reason did not complete: EmptyCompletionError, unmarked, so it is a request failure
      (retried, and flagged if it persists).
    - Otherwise the server completed the generation, and an unusable reply is the model's own move
      (marked as model output: played as 'none', not retried, rated): a structured-output refusal
      (content None, ``refusal`` set; ModelRefusalError keeps the text), or an empty or whitespace-only
      reply whatever the finish reason ('stop', 'length' at the token limit, 'content_filter', ...).
    """
    choices = getattr(response, "choices", None)
    if not choices or getattr(choices[0], "message", None) is None:
        raise EmptyCompletionError(f"no choices in the completion from {model_name}", finish_reason=None)
    choice = choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason in SERVER_ABORT_FINISH_REASONS:
        raise EmptyCompletionError(
            f"the server aborted the generation from {model_name} (finish_reason={finish_reason!r})",
            finish_reason=finish_reason,
        )
    if isinstance(content, str) and content.strip():
        return content
    refusal = getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        refused = ModelRefusalError(
            f"structured-output refusal from {model_name} (finish_reason={finish_reason!r}): {refusal}",
            refusal=refusal,
            finish_reason=finish_reason,
        )
        _mark_model_output_error(refused)
        raise refused
    exc = EmptyCompletionError(
        f"empty completion from {model_name} (finish_reason={finish_reason!r})",
        finish_reason=finish_reason,
    )
    if finish_reason is not None:
        _mark_model_output_error(exc)
    raise exc


_MODEL_LOADING_WORDS = (
    "no models loaded",
    "unloaded",
    "not loaded",
    "model_not_found",
    "crash",
    "channel error",
    "temporarily",
)


def _model_loading_error(e: BaseException) -> bool:
    """A self-hosted server's message that its model is loading, unloaded or crashed (it may come with a 4xx)."""
    msg = str(e).lower()
    return any(k in msg for k in _MODEL_LOADING_WORDS)


def _transient_endpoint_error(e: Exception) -> bool:
    """Errors worth waiting out: a model server that is down or reloading, a dropped connection. Used only
    for failed requests without an HTTP status (see is_retryable_request_error)."""
    msg = str(e).lower()
    return any(
        k in msg
        for k in (
            "404",
            "not found",
            "connection",
            "connect",
            "timed out",
            "timeout",
            "502",
            "503",
            "500",
            "server disconnected",
            "temporarily",
            "crash",
            "channel error",
            "model_not_found",
            "no models loaded",
            "unloaded",
            "not loaded",
            "empty completion",
        )
    )


@gin.configurable
@validate_call
async def agenerate_action(
    model_name: str,
    history: str,
    turn_number: int,
    action_types: list[ActionType],
    agent: str,
    goal: str,
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    script_like: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    strict_action_constraint: bool = False,
    custom_template: str | None = None,
    return_prompt_and_response: bool = False,
    structured_output: bool = True,
) -> AgentAction | tuple[AgentAction, list[dict[str, str]], str]:
    """
    Using langchain to generate an example episode
    """
    try:
        if custom_template:
            if script_like:
                raise ValueError(
                    "script_like and custom_template are mutually exclusive"
                )
            template = custom_template
        elif script_like:
            # model as playwright
            template = """
                Now you are a famous playwright, your task is to continue writing one turn for agent {agent} under a given background and history to help {agent} reach social goal. Please continue the script based on the previous turns. You can only generate one turn at a time.
                You can find {agent}'s background and goal in the 'Here is the context of the interaction' field.
                You should try your best to achieve {agent}'s goal in a way that align with their character traits.
                Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
                {history}.
                The script has proceeded to Turn #{turn_number}. Current available action types are
                {action_list}.
                Note: The script can be ended if 1. one agent have achieved social goals, 2. this conversation makes the agent uncomfortable, 3. the agent find it uninteresting/you lose your patience, 4. or for other reasons you think it should stop.

                Please only generate a JSON string including the action type and the argument.
                Your action should follow the given format:
                {format_instructions}
            """
        else:
            # Normal case, model as agent
            template = """
                Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
                You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
                Note that {agent}'s goal is only visible to you.
                You should try your best to achieve {agent}'s goal in a way that align with their character traits.
                Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
                {history}.
                You are at Turn #{turn_number}. Your available action types are
                {action_list}.
                Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

                Please only generate a JSON string including the action type and the argument.
                Your action should follow the given format:
                {format_instructions}
            """

        # Create dynamic AgentAction model with restricted ActionType
        if strict_action_constraint and action_types:
            # Create a dynamic Literal for the allowed action types
            # Use __getitem__ to dynamically create Literal from list of strings
            DynamicActionType = Literal.__getitem__(tuple(action_types))

            # Create a dynamic Pydantic model
            from pydantic import create_model, Field

            DynamicAgentAction = create_model(
                "AgentAction",
                action_type=(
                    DynamicActionType,
                    Field(
                        ...,
                        description="whether to speak at this turn or choose to not do anything",
                    ),
                ),
                argument=(
                    str,
                    Field(
                        ...,
                        description="the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action",
                    ),
                ),
                __base__=AgentAction,
            )

            output_parser_obj: PydanticOutputParser[Any] = PydanticOutputParser(
                pydantic_object=DynamicAgentAction
            )
        else:
            output_parser_obj = PydanticOutputParser(pydantic_object=AgentAction)

        async def _call() -> Any:
            return await agenerate(
                model_name=model_name,
                template=template,
                input_values=dict(
                    agent=agent,
                    turn_number=str(turn_number),
                    history=history,
                    action_list=" ".join(action_types),
                    goal=goal,
                ),
                output_parser=output_parser_obj,
                temperature=temperature,
                structured_output=structured_output,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
                return_prompt_and_response=return_prompt_and_response,
                repair_bad_output=False,  # a malformed reply is the model's move, never repaired
            )

        # Retry transient request failures (a dropped connection, a timeout, a server error, a
        # self-hosted model that crashed and is being reloaded) with exponential backoff, on by
        # default: SOTOPIA_ENDPOINT_RETRIES (default 6) retries, the first after
        # SOTOPIA_ENDPOINT_RETRY_WAIT seconds (default 5), doubling up to
        # SOTOPIA_ENDPOINT_RETRY_MAX_WAIT (default 60). A malformed model reply is not retried.
        retries = int(_env_number("SOTOPIA_ENDPOINT_RETRIES", DEFAULT_ENDPOINT_RETRIES))
        retry_errors: list[dict[str, str]] = []
        for attempt in range(retries + 1):
            try:
                result = await _call()
                meta = LAST_COMPLETION_METADATA.get() or {}
                if retry_errors:
                    meta = dict(meta)
                    meta["request_errors"] = retry_errors
                    LAST_COMPLETION_METADATA.set(meta)
                log_progress_event(
                    "call", model=model_name, ok=True, elapsed=meta.get("elapsed_seconds"), attempt=attempt
                )
                return result
            except Exception as e:
                if attempt < retries and is_retryable_request_error(e):
                    wait = _retry_wait(attempt)
                    retry_errors.append({"type": type(e).__name__, "message": str(e)[:500]})
                    log.warning(
                        f"Endpoint error for {model_name} ({str(e)[:160]}); retry {attempt + 1}/{retries} in {wait:.0f}s"
                    )
                    log_progress_event("call", model=model_name, ok=False, retry=True, error=str(e)[:200])
                    await asyncio.sleep(wait)
                    continue
                if retry_errors:
                    setattr(e, "_sotopia_retry_errors", retry_errors)
                raise
    except Exception as e:
        # The API refused the response schema itself (or the model has none): play the turn without one.
        # Only a request error about the schema qualifies (is_schema_rejection), never text in a reply; the
        # unstructured call does not repair a malformed reply either (repair_bad_output=False above).
        if structured_output and is_schema_rejection(e):
            log.info(f"Retrying without structured_output for model {model_name}")
            previous_metadata = LAST_COMPLETION_METADATA.get() or {}
            LAST_COMPLETION_METADATA.set(None)
            fallback_result = await agenerate_action(
                model_name=model_name,
                history=history,
                turn_number=turn_number,
                action_types=action_types,
                agent=agent,
                goal=goal,
                temperature=temperature,
                script_like=script_like,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
                strict_action_constraint=strict_action_constraint,
                custom_template=custom_template,
                return_prompt_and_response=return_prompt_and_response,
                structured_output=False,
            )
            metadata = dict(LAST_COMPLETION_METADATA.get() or {})
            attempts = previous_metadata.get("attempts", []) + metadata.get(
                "attempts", []
            )
            metadata.update(
                structured_output_fallback=True,
                structured_output_error={"type": type(e).__name__, "message": str(e)[:500]},
                attempts=attempts,
                elapsed_seconds=sum(attempt["elapsed_seconds"] for attempt in attempts),
            )
            LAST_COMPLETION_METADATA.set(metadata)
            return fallback_result
        log.warning(f"Failed to generate action due to {e}")
        log_progress_event("call", model=model_name, ok=False, error=str(e)[:200])
        metadata = dict(LAST_COMPLETION_METADATA.get() or {})
        metadata["action_error"] = {"type": type(e).__name__, "message": str(e)}
        # The turn is still played as 'none' so the episode runs on. A model's own reply is its move
        # (request_failed False, rated). Anything else is flagged: calculate_elo drops such episodes, and
        # run_elo_tournament replays their rosters on resume unless the failure is permanent (a 400 or a
        # context-window error, which would fail again the same way).
        model_move = is_model_output_error(e)
        metadata["request_failed"] = not model_move
        if not model_move and is_permanent_request_error(e):
            metadata["permanent_failure"] = True
        if isinstance(e, ModelRefusalError):
            metadata["refusal"] = e.refusal
        if getattr(e, "_sotopia_retry_errors", None):
            metadata["request_errors"] = getattr(e, "_sotopia_retry_errors")
        LAST_COMPLETION_METADATA.set(metadata)
        action = AgentAction(action_type="none", argument="")
        if return_prompt_and_response:
            if model_move:
                # Keep the model's raw reply (and the prompt) in the log; the error is in action_error.
                return action, getattr(e, "_sotopia_prompt", []), getattr(e, "_sotopia_raw_reply", "")
            return action, [], str(e)
        return action


@gin.configurable
@validate_call
async def agenerate_script(
    model_name: str,
    background: ScriptBackground,
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    agent_names: list[str] = [],
    agent_name: str = "",
    history: str = "",
    single_step: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> tuple[ScriptInteractionReturnType, str]:
    """
    Using langchain to generate an the script interactions between two agent
    The script interaction is generated in a single generation process.
    Note that in this case we do not require a json format response,
    so the failure rate will be higher, and it is recommended to use at least llama-2-70b.
    """
    try:
        if single_step:
            result = await agenerate(
                model_name=model_name,
                template="""Now you are a famous playwright, your task is to continue writing one turn for agent {agent} under a given background and history to help {agent} reach social goal. Please continue the script based on the previous turns. You can only generate one turn at a time.

                Here are the conversation background and history:
                {background}
                {history}

                Remember that you are an independent scriptwriter and should finish the script by yourself.
                The output should only contain the script following the format instructions, with no additional comments or text.

                Here are the format instructions:
                {format_instructions}""",
                input_values=dict(
                    background=background.to_natural_language(),
                    history=history,
                    agent=agent_name,
                ),
                output_parser=ScriptOutputParser(
                    agent_names=agent_names,
                    background=background.to_natural_language(),
                    single_turn=True,
                ),
                temperature=temperature,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )
            return cast(tuple[ScriptInteractionReturnType, str], result)

        else:
            result = await agenerate(
                model_name=model_name,
                template="""
                Please write the script between two characters based on their social goals with a maximum of 20 turns.

                {background}
                Your action should follow the given format:
                {format_instructions}
                Remember that you are an independent scriptwriter and should finish the script by yourself.
                The output should only contain the script following the format instructions, with no additional comments or text.""",
                input_values=dict(
                    background=background.to_natural_language(),
                ),
                output_parser=ScriptOutputParser(
                    agent_names=agent_names,
                    background=background.to_natural_language(),
                    single_turn=False,
                ),
                temperature=temperature,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )
            return cast(tuple[ScriptInteractionReturnType, str], result)
    except Exception as e:
        # TODO raise(e) # Maybe we do not want to return anything?
        print(f"Exception in agenerate {e}")
        return_default_value: ScriptInteractionReturnType = (
            ScriptInteraction.default_value_for_return_type()
        )
        return (return_default_value, "")


@validate_call
def process_history(
    script: ScriptBackground | EnvResponse | dict[str, AgentAction],
) -> str:
    """
    Format the script background
    """
    result = ""
    if isinstance(script, ScriptBackground | EnvResponse):
        script = script.dict()
        result = "The initial observation\n\n"
    for key, value in script.items():
        if value:
            result += f"{key}: {value} \n"
    return result


@validate_call
async def agenerate_init_profile(
    model_name: str,
    basic_info: dict[str, str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template="""Please expand a fictional background for {name}. Here is the basic information:
            {name}'s age: {age}
            {name}'s gender identity: {gender_identity}
            {name}'s pronouns: {pronoun}
            {name}'s occupation: {occupation}
            {name}'s big 5 personality traits: {bigfive}
            {name}'s moral Foundation: think {mft} is more important than others
            {name}'s Schwartz portrait value: {schwartz}
            {name}'s decision-making style: {decision_style}
            {name}'s secret: {secret}
            Include the previous information in the background.
            Then expand the personal backgrounds with concrete details (e.g, look, family, hobbies, friends and etc.)
            For the personality and values (e.g., MBTI, moral foundation, and etc.),
            remember to use examples and behaviors in the person's life to demonstrate it.
            """,
        input_values=dict(
            name=basic_info["name"],
            age=basic_info["age"],
            gender_identity=basic_info["gender_identity"],
            pronoun=basic_info["pronoun"],
            occupation=basic_info["occupation"],
            bigfive=basic_info["Big_Five_Personality"],
            mft=basic_info["Moral_Foundation"],
            schwartz=basic_info["Schwartz_Portrait_Value"],
            decision_style=basic_info["Decision_making_Style"],
            secret=basic_info["secret"],
        ),
        output_parser=StrOutputParser(),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@validate_call
async def convert_narratives(
    model_name: str,
    narrative: str,
    text: str,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    if narrative == "first":
        return await agenerate(
            model_name=model_name,
            template="""Please convert the following text into a first-person narrative.
            e.g, replace name, he, she, him, her, his, and hers with I, me, my, and mine.
            {text}""",
            input_values=dict(text=text),
            output_parser=StrOutputParser(),
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
        )
    elif narrative == "second":
        return await agenerate(
            model_name=model_name,
            template="""Please convert the following text into a second-person narrative.
            e.g, replace name, he, she, him, her, his, and hers with you, your, and yours.
            {text}""",
            input_values=dict(text=text),
            output_parser=StrOutputParser(),
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
        )
    else:
        raise ValueError(f"Narrative {narrative} is not supported.")


@validate_call
async def agenerate_goal(
    model_name: str,
    background: str,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template="""Please generate your goal based on the background:
            {background}
            """,
        input_values=dict(background=background),
        output_parser=StrOutputParser(),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )

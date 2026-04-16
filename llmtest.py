"""Small llmlib demo: text completion + local time tool."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from llmlib.llm import OpenAI, Tool, ToolCall, ToolExecutor


def get_current_time(timezone_name: str = "UTC") -> str:
    """Return current time in ISO format.

    Only UTC is supported in this minimal demo.
    """
    if timezone_name.upper() != "UTC":
        return "Only UTC is supported in this demo. Use timezone_name='UTC'."
    now = datetime.now(timezone.utc)
    return now.isoformat()


def main() -> None:
    base_url = os.getenv("LLM_API_URL", "http://127.0.0.1:1234/v1")
    model = os.getenv("LLM_MODEL", "qwen3.5-4b")
    api_key = os.getenv("LLM_API_KEY", "")

    client = OpenAI(base_url=base_url, model=model, api_key=api_key)

    tool = Tool(
        name="get_current_time",
        description="Return current UTC time in ISO format.",
        parameters={
            "type": "object",
            "properties": {
                "timezone_name": {
                    "type": "string",
                    "description": "Timezone name. Use 'UTC'.",
                }
            },
            "required": ["timezone_name"],
            "additionalProperties": False,
        },
    )

    executor = ToolExecutor()
    executor.register("get_current_time", get_current_time)

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Eres un asistente breve y útil. "
                "Debes completar el texto del usuario en español. "
                "Si necesitas hora actual, llama la herramienta get_current_time."
            ),
        },
        {
            "role": "user",
            "content": (
                "Completa este texto en una sola frase: "
                "'Ahora mismo son las __ y por eso podemos decir que...' "
                "Si necesitas hora actual, usa la tool get_current_time con timezone_name='UTC'."
            ),
        },
    ]

    try:
        first = client.chat.completions.create(
            messages=messages,
            tools=[tool],
            tool_choice="auto",
            max_tokens=200,
        )

        choice = first.choices[0]

        def extract_text(current_choice) -> str:
            text = (current_choice.message.content or "").strip()
            if text:
                return text
            return "[sin contenido en la respuesta del modelo]"

        if choice.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": choice.message.content,
                    "tool_calls": choice.tool_calls,
                }
            )

            parsed_calls = [ToolCall.from_dict(tc) for tc in choice.tool_calls]
            for result in executor.execute_all(parsed_calls):
                messages.append(result.to_message())

            second = client.chat.completions.create(
                messages=messages,
                tools=[tool],
                max_tokens=200,
            )
            final_text = extract_text(second.choices[0])
            if final_text == "[sin contenido en la respuesta del modelo]":
                current_time = get_current_time("UTC")
                final_text = (
                    f"Ahora mismo son las {current_time} y por eso podemos decir que "
                    "el tiempo en UTC sigue avanzando con precisión."
                )
            print(final_text)
        else:
            # Fallback: if model does not emit tool_calls, inject current time manually
            # and ask for final completion so output is never blank/useless.
            current_time = get_current_time("UTC")
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "No hubo tool_calls. Usa este dato de hora UTC para completar: "
                        f"{current_time}"
                    ),
                }
            )
            fallback = client.chat.completions.create(
                messages=messages,
                max_tokens=120,
            )
            fallback_text = extract_text(fallback.choices[0])
            if fallback_text == "[sin contenido en la respuesta del modelo]":
                fallback_text = (
                    f"Ahora mismo son las {current_time} y por eso podemos decir que "
                    "tenemos una referencia temporal clara para completar la frase."
                )
            print(fallback_text)
    finally:
        client.close()


if __name__ == "__main__":
    main()

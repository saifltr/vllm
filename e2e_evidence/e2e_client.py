"""Send tool-calling requests to a running vLLM server and print what comes back."""
import json, sys
from openai import OpenAI

port = sys.argv[1] if len(sys.argv) > 1 else "8123"
label = sys.argv[2] if len(sys.argv) > 2 else ""
client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY")
model = client.models.list().data[0].id

TOOLS = [{"type": "function", "function": {"name": "set_alarm", "description": "Set an alarm.", "parameters": {
    "type": "object",
    "properties": {
        "hour": {"type": "integer", "description": "Hour of the day, 0-23"},
        "enabled": {"type": "boolean", "description": "Whether the alarm is on"},
        "label": {"type": "string", "description": "Alarm label"},
        "days": {"type": "array", "items": {"type": "string"}, "description": "Days of the week"},
    },
    "required": ["hour", "enabled", "label", "days"]}}}]
PROMPT = "Set an alarm for 7 o'clock labeled Wake up, enabled, on mon and tue."

WEATHER_TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather.", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
    "required": ["city", "unit"]}}}]
WEATHER_PROMPT = "What's the weather in San Francisco in celsius?"

def run(stream, tools=TOOLS, prompt=PROMPT):
    kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}], tools=tools, tool_choice="auto",
                  temperature=0, max_tokens=128, stream=stream)
    if not stream:
        r = client.chat.completions.create(**kwargs)
        m = r.choices[0].message
        calls = [(c.function.name, c.function.arguments) for c in (m.tool_calls or [])]
        return m.content, calls, r.choices[0].finish_reason
    acc = {}; content = ""; finish = None
    for chunk in client.chat.completions.create(**kwargs):
        ch = chunk.choices[0]
        if ch.finish_reason: finish = ch.finish_reason
        if ch.delta.content: content += ch.delta.content
        for tc in ch.delta.tool_calls or []:
            e = acc.setdefault(tc.index, {"name": None, "args": ""})
            if tc.function.name: e["name"] = tc.function.name
            e["args"] += tc.function.arguments or ""
    return content, [(e["name"], e["args"]) for _, e in sorted(acc.items())], finish

for scenario, tools, prompt in (("set_alarm (int/bool/array/string)", TOOLS, PROMPT), ("get_weather (two strings)", WEATHER_TOOLS, WEATHER_PROMPT)):
  for stream in (False, True):
    content, calls, finish = run(stream, tools, prompt)
    print(f"### {label} {scenario} stream={stream} finish_reason={finish}")
    print(f"content: {content!r}")
    for name, args in calls:
        try: parsed = json.loads(args); ok = "valid JSON"
        except Exception as e: parsed = None; ok = f"INVALID JSON ({e})"
        print(f"tool_call: {name} arguments={args!r}  -> {ok}")
        if parsed is not None: print(f"  parsed: {json.dumps(parsed, ensure_ascii=False)}")
    print()

# Raw model text for the same request, via /v1/completions on the rendered prompt.
if label == "main":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("unsloth/functiongemma-270m-it")
    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tools=TOOLS, tokenize=False, add_generation_prompt=True)
    r = client.completions.create(model=model, prompt=prompt, temperature=0, max_tokens=128,
                                  extra_body={"skip_special_tokens": False})
    print("### raw model output (/v1/completions, skip_special_tokens=false)")
    print(repr(r.choices[0].text))

[SYSTEM]
Reply in plain text. No LaTeX, no $ symbols.

[USER]
What is 23 * 7, and how many letters are in the word 'automation'?
------------------------------------------------------------
TOOLS YOU MAY CALL:
- calculator: Evaluate a math expression, e.g. '23*7+1'.
  args schema: {"additionalProperties": false, "properties": {"expr": {"type": "string"}}, "required": ["expr"], "type": "object"}
- word_length: Count the characters in a word.
  args schema: {"additionalProperties": false, "properties": {"word": {"type": "string"}}, "required": ["word"], "type": "object"}

Reply with EXACTLY ONE JSON object and nothing else (no prose, no code fence).
  to call a tool : {"tool": "<name>", "args": { ... }}
  several at once: {"tools": [{"tool":"<name>","args":{...}}, ...]}
  to answer      : {"final": "<your answer to the user>"}
------------------------------------------------------------

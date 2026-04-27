from __future__ import annotations

from schema import NodeSchema


def extract_property_info(node_schema: NodeSchema, prop_name: str) -> dict:
    prop_obj = None
    props = getattr(node_schema, "properties", None) or getattr(node_schema, "Props", None) or []

    if isinstance(props, dict):
        prop_obj = props.get(prop_name)
    else:
        for p in props:
            pname = getattr(p, "name", None) or getattr(p, "Name", None)
            if pname == prop_name:
                prop_obj = p
                break

    return {
        "node_name": getattr(node_schema, "name", "") or getattr(node_schema, "Name", ""),
        "node_description": getattr(node_schema, "description", "") or getattr(node_schema, "Desc", ""),
        "property_name": prop_name,
        "property_description": getattr(prop_obj, "description", "") if prop_obj else "",
        "property_type": getattr(prop_obj, "type", "") if prop_obj else "",
        "required": getattr(prop_obj, "required", "") if prop_obj else "",
        "enum": getattr(prop_obj, "enum", []) if prop_obj else [],
    }


# def build_textual_relation_messages(
#     node_schema,
#     prop_a_label: str,
#     prop_b_label: str,
#     prop_a_values: list[str],
#     prop_b_values: list[str],
# ) -> list[dict[str, str]]:
#     prop_a_info = extract_property_info(node_schema, prop_a_label)
#     prop_b_info = extract_property_info(node_schema, prop_b_label)

#     system_msg = (
#         "You are a biomedical data quality assistant.\n\n"
#         "Hard rules:\n"
#         "- Use ONLY the information provided in the prompt.\n"
#         "- Do NOT invent facts, labels, properties, or biological relationships that are not supported by the input.\n"
#         "- Do NOT rely on outside knowledge unless it is directly supported by the provided text.\n"
#         '- If the relation is unclear, weak, ambiguous, or unsupported, return "unrelated".\n'
#         "- Return ONLY valid JSON.\n"
#         "- Do not include markdown, bullets, code fences, or extra commentary.\n"
#         '- Output must match exactly: {"label": "...", "reason": "..."}.\n\n'
#         "Allowed labels:\n"
#         "- matching\n"
#         "- related\n"
#         "- complementary\n"
#         "- unrelated\n\n"
#         "Label rules:\n"
#         "- matching: same or near-same meaning\n"
#         "- related: same biological context, but not identical\n"
#         "- complementary: different but biologically supportive/compatible\n"
#         "- unrelated: no meaningful relation\n"
#     )

#     user_msg = (
#         "Analyze the relationship between two properties.\n\n"
#         "PROPERTY 1:\n"
#         f"Node name: {prop_a_info['node_name']}\n"
#         f"Node description: {prop_a_info['node_description']}\n"
#         f"Property name: {prop_a_info['property_name']}\n"
#         f"Property description: {prop_a_info['property_description']}\n"
#         f"Property type: {prop_a_info['property_type']}\n"
#         f"Required: {prop_a_info['required']}\n"
#         f"Enum: {prop_a_info['enum']}\n"
#         f"Example values: {prop_a_values}\n\n"
#         "PROPERTY 2:\n"
#         f"Node name: {prop_b_info['node_name']}\n"
#         f"Node description: {prop_b_info['node_description']}\n"
#         f"Property name: {prop_b_info['property_name']}\n"
#         f"Property description: {prop_b_info['property_description']}\n"
#         f"Property type: {prop_b_info['property_type']}\n"
#         f"Required: {prop_b_info['required']}\n"
#         f"Enum: {prop_b_info['enum']}\n"
#         f"Example values: {prop_b_values}\n\n"
#         "Classify the relation between Property 1 and Property 2.\n"
#         'Return only the JSON object: {"label": "...", "reason": "..."}'
#     )

#     return [
#         {"role": "system", "content": system_msg},
#         {"role": "user", "content": user_msg},
#     ]

def build_textual_relation_messages(
    node_schema,
    prop_a_label: str,
    prop_b_label: str,
    prop_a_values: list[str],
    prop_b_values: list[str],
    model: str = "openai",  # <-- add this
):
    prop_a_info = extract_property_info(node_schema, prop_a_label)
    prop_b_info = extract_property_info(node_schema, prop_b_label)

    if "gemini" in model.lower():
        # ✅ Gemini prompt (single string)
        return f"""
You are a biomedical data quality assistant.

Hard rules:
- Use ONLY the information provided below.
- Do NOT invent facts, labels, properties, or biological relationships.
- Do NOT rely on outside knowledge unless explicitly supported.
- If the relation is unclear, weak, ambiguous, or unsupported, return "unrelated".
- Return ONLY valid JSON.
- Do not include markdown, bullets, code fences, or extra commentary.
- Output must match exactly: {{"label": "...", "reason": "..."}}.

Allowed labels:
- matching
- related
- complementary
- unrelated

Label rules:
- matching: same or near-same meaning
- related: same biological context, but not identical
- complementary: different but biologically supportive/compatible
- unrelated: no meaningful relation

---

Analyze the relationship between two properties.

PROPERTY 1:
Node name: {prop_a_info['node_name']}
Node description: {prop_a_info['node_description']}
Property name: {prop_a_info['property_name']}
Property description: {prop_a_info['property_description']}
Property type: {prop_a_info['property_type']}
Required: {prop_a_info['required']}
Enum: {prop_a_info['enum']}
Example values: {prop_a_values}

PROPERTY 2:
Node name: {prop_b_info['node_name']}
Node description: {prop_b_info['node_description']}
Property name: {prop_b_info['property_name']}
Property description: {prop_b_info['property_description']}
Property type: {prop_b_info['property_type']}
Required: {prop_b_info['required']}
Enum: {prop_b_info['enum']}
Example values: {prop_b_values}

---

Classify the relation between Property 1 and Property 2.

Return only the JSON object:
{{"label": "...", "reason": "..."}}
""".strip()

    # ✅ Default (OpenAI-style messages)
    system_msg = (
        "You are a biomedical data quality assistant.\n\n"
        "Hard rules:\n"
        "- Use ONLY the information provided in the prompt.\n"
        "- Do NOT invent facts, labels, properties, or biological relationships that are not supported by the input.\n"
        "- Do NOT rely on outside knowledge unless it is directly supported by the provided text.\n"
        '- If the relation is unclear, weak, ambiguous, or unsupported, return "unrelated".\n'
        "- Return ONLY valid JSON.\n"
        "- Do not include markdown, bullets, code fences, or extra commentary.\n"
        '- Output must match exactly: {"label": "...", "reason": "..."}.\n\n'
        "Allowed labels:\n"
        "- matching\n"
        "- related\n"
        "- complementary\n"
        "- unrelated\n\n"
        "Label rules:\n"
        "- matching: same or near-same meaning\n"
        "- related: same biological context, but not identical\n"
        "- complementary: different but biologically supportive/compatible\n"
        "- unrelated: no meaningful relation\n"
    )

    user_msg = (
        "Analyze the relationship between two properties.\n\n"
        "PROPERTY 1:\n"
        f"Node name: {prop_a_info['node_name']}\n"
        f"Node description: {prop_a_info['node_description']}\n"
        f"Property name: {prop_a_info['property_name']}\n"
        f"Property description: {prop_a_info['property_description']}\n"
        f"Property type: {prop_a_info['property_type']}\n"
        f"Required: {prop_a_info['required']}\n"
        f"Enum: {prop_a_info['enum']}\n"
        f"Example values: {prop_a_values}\n\n"
        "PROPERTY 2:\n"
        f"Node name: {prop_b_info['node_name']}\n"
        f"Node description: {prop_b_info['node_description']}\n"
        f"Property name: {prop_b_info['property_name']}\n"
        f"Property description: {prop_b_info['property_description']}\n"
        f"Property type: {prop_b_info['property_type']}\n"
        f"Required: {prop_b_info['required']}\n"
        f"Enum: {prop_b_info['enum']}\n"
        f"Example values: {prop_b_values}\n\n"
        "Classify the relation between Property 1 and Property 2.\n"
        'Return only the JSON object: {"label": "...", "reason": "..."}'
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
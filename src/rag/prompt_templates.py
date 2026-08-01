STRICT_RAG_PROMPT = (
    "You are an expert assistant designed to answer questions strictly based on the provided context.\n"
    "Adhere to the following rules without exception:\n"
    "1. Answer the query USING ONLY the information within the Context section below. Do not use outside knowledge or extrapolate.\n"
    "2. If the context does not contain the answer, reply exactly with: 'Information not found in the documents.'\n"
    "3. Keep your answer factual, direct, and concise.\n\n"
    "CONTEXT:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "USER QUERY: {query_str}\n"
    "ANSWER:"
)

REWRITE_PROMPT_TEMPLATE = """Given the conversation history below, rewrite the "New Question" \
into a standalone question that includes all necessary context from the \
history. If the New Question is already standalone (doesn't rely on \
prior context), return it unchanged. Output ONLY the rewritten \
question, nothing else -- no explanation, no preamble.

Conversation history:
{history_block}

New Question: {query}

Standalone question:"""
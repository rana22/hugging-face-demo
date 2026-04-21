# Record = {
#     "id": str,

#     # All extracted entities (normalized)
#     "entities": {
#         "gene": set(),
#         "disease": set(),
#         "pathway": set(),
#         "protein": set(),
#         "cell_type": set(),
#         "tissue": set(),
#         "organism": set(),
#         "intervention": set(),
#         "phenotype": set(),
#         "other": set()
#     },

#     # Context-aware signals
#     "relations": [
#         # (entity1, relation, entity2)
#         ("TP53", "involved_in", "apoptosis"),
#         ("EMT", "associated_with", "invasion"),
#     ],

#     # Embeddings
#     "text_embedding": vector,

#     # Optional: per-field embeddings
#     "field_embeddings": [vector, vector, ...]

#     # Metadata (optional)
#     "raw_text": full_concatenated_text
# }

# 2. Entity Types Support (v1)

# Start with these (they cover ~90% of biomedical cases):

# Type	Examples
# gene	TP53, KRAS
# disease	bladder cancer
# pathway	EMT, apoptosis
# protein	p53 protein
# cell_type	epithelial cell
# tissue	bladder epithelium
# organism	human, canine
# intervention	immunotherapy
# phenotype	invasion, metastasis
# other	fallback


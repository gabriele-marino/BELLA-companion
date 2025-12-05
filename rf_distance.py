import dendropy

tns = dendropy.TaxonNamespace()

# Load two nexus trees
t1 = dendropy.Tree.get(
    path="outputs/runs/eucovid/flights_over_population/BELLA/TypedNodeTrees.annotated.trees",
    schema="nexus",
    taxon_namespace=tns,
)
t2 = dendropy.Tree.get(
    path="outputs/runs/eucovid/flights_over_population/GLM/TypedNodeTrees.annotated.trees",
    schema="nexus",
    taxon_namespace=tns,
)

# Compute unweighted RF distance
rf = dendropy.calculate.treecompare.symmetric_difference(t1, t2)

# If you also want the normalized version:
# max_rf = t1.max_robinson_foulds_distance(t2)
# normalized = rf / max_rf

print("RF distance:", rf)
# print("Normalized RF distance:", normalized)

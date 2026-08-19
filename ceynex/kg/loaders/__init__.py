"""Knowledge graph loaders. One module per owner.

`trade_agreements` is M2's — both sector loaders attach `COVERED_BY` edges to
those nodes, so neither M1 nor M3 should own them. `agriculture` is M1's and
`apparel` is M3's; they arrive in their own repos or as their own modules here.

Every loader statement must be `MERGE`, never `CREATE`. Three people load into
one graph and re-run their loaders freely.
"""

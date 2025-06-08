# md -> ipynb
uv run jupytext --to notebook --pipe black src/querying_data/notebooks/querying.md --output src/querying_data/notebooks/querying.ipynb

# ipynb -> md
uv run jupytext --to markdown --pipe black src/querying_data/notebooks/querying.ipynb --output src/querying_data/notebooks/querying.md

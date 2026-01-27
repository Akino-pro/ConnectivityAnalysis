from pathlib import Path
p = Path("./data/4R_atlas.pkl")
print("exists:", p.exists())
print("abs:", p.resolve())
print("size:", p.stat().st_size)

with p.open("rb") as f:
    head = f.read(32)
print("head bytes:", head)
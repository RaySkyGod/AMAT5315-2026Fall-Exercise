# week2/Makefile — named entry points for the md crate's CLI (learning sheet,
# Part 4: "A Makefile gives commands names").

# Contract run: the crate's default flags in release mode, writing run.json
# and traj.jsonl into week2/artifacts/. Equivalent to the full form
#   md run --n 100 --rho 0.8 --temperature 0.5 --dt 0.01 \
#          --eq-steps 2000 --steps 10000 --sample-every 50 --seed 2026
# because every flag defaults to the contract run.
reproduce:
	cargo run --manifest-path md/Cargo.toml --release -- run --out artifacts

.PHONY: reproduce

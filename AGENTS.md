# AGENTS.md

Guidance for coding agents working in this repository.

## Owner

I am Mengbo Guo, a student enrolled in AMAT5315 (Fall 2026) — an introductory
course on computational methods for physical systems such as quantum physics
and spin glasses. The course assumes knowledge of linear algebra and
familiarity with at least one programming language, and covers
matrix/tensor computation, differential programming, combinatorial
optimization, CUDA programming, and the Julia programming language.

## What this repository is for

This repository holds my weekly exercises for the course, organized one
directory per week: `week1/`, `week2/`, and so on as the semester progresses.
It is purely personal coursework and exercise practice — not a portfolio or
collaborative project.

Current contents:

- `week1/` — Python exercises (numerical estimation of π); tested with `pytest`.
- `week2/` — Rust molecular-dynamics exercises in `week2/md/` (tested with
  `cargo test`), plus Python plotting scripts and generated figures.

## Working conventions

- Work test-first: write failing tests, then implement until they pass.
- Run the tests for the week you touch: `pytest` for Python weeks,
  `cargo test` for Rust crates.
- Keep plotting scripts separate from core implementation code, as in `week2/`.
- Commit messages use a `weekN: summary` prefix for week-specific work, e.g.
  `week2: plot LJ pair energy (color) and force (arrows) around one atom`.

Memory probe: W1-MEMORY-5315

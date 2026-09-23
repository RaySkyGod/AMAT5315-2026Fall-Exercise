---
name: tutor
description: "Turn a lesson from a local file or web address into a guided tutoring session. Use when the user asks to be tutored on, study, or review a lesson, e.g. a weekly learning sheet from AMAT5315 (local .md, or a PDF from https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/)."
---

# Tutor

Run a guided tutoring session from a lesson source: a local file, a URL, or a
bare week number.

## Resolving the lesson source

1. **Local path** — read the file directly (any text format: `.md`, `.txt`).
2. **URL** — download it first. If it ends in `.pdf` (or serves
   `Content-Type: application/pdf`), treat it as a PDF (next step).
3. **Bare week number** like `week2` — resolve to
   `https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/week2-learning-sheet.pdf`
   and treat it as a PDF.

## Handling PDFs

Before tutoring on a PDF you MUST extract its text with the `pypdf` package:

1. **Setup (once):** if `python3 -c "import pypdf"` fails, run
   `pip install --user pypdf` first.
2. **Download:** save the PDF next to the existing week's materials if the
   directory exists (e.g. `week2/week2-learning-sheet.pdf`), otherwise to
   `/tmp/`. Skip the download if the file is already there.
3. **Extract text** with pypdf, e.g.:

   ```bash
   python3 -c "from pypdf import PdfReader; print('\n\n'.join(p.extract_text() for p in PdfReader('LESSON.pdf').pages))"
   ```

4. Tutor from the extracted text. Note to the learner when extraction looks
   garbled (equations/figures may not survive) and fall back to summarizing
   section structure.

## Session format

Guided walkthrough with comprehension checks:

1. **Orient** — state the lesson title and list its sections; ask the learner
   which parts they want to focus on (or offer "all of it").
2. **Walk through each section** — explain intuitively first, formal
   derivations only on request. After each section, ask 1–2 short check
   questions; correct misconceptions Socratically before moving on.
3. **Connect** — when the lesson relates to code already in this repo's
   `weekN/` directories, point out the connection (e.g. the LJ potential in
   `week2/md/`).
4. **Recap** — end with a 3–5 bullet summary of the key ideas and the
   learner's weak spots to revisit.

Stay conceptual unless the learner asks to write code or drill problems.
Keep the session purely conversational — do not write notes files unless asked.

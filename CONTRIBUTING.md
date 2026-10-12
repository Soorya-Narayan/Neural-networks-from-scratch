# Contributing to Neural Networks from Scratch

Thank you for your interest in contributing to **Neural Networks from Scratch**!

This project is built as a step-by-step textbook and code repository designed to teach deep learning from first principles. Because clarity and pedagogical value are our primary goals, we prioritize clean, intuitive code and thorough explanations over clever shortcuts.

---

## Guiding Principles

1. **Clarity Over Cleverness**: Code is meant to be read by learners. Avoid esoteric one-liners or premature abstractions.
2. **Intuition Before Equations**: Always explain *why* something works geometrically or conceptually before writing equations.
3. **Every Equation Needs Code**: Whenever mathematical notation is introduced, provide the exact lines of code that implement it.
4. **Verified Correctness**: Any new analytical gradient or layer must be validated with:
   - Numerical gradient checking (finite difference approximation).
   - PyTorch autograd reference comparison.
5. **Lightweight & Portable**: Everything should run comfortably on a standard laptop (CPU or Apple Silicon MPS) without requiring high-end GPUs or massive datasets.

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Soorya-Narayan/Neural-networks-from-scratch.git
   cd Neural-networks-from-scratch
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests**:
   ```bash
   pytest
   ```

---

## Submitting Pull Requests

1. **Create a branch**: `git checkout -b feature/my-new-chapter-or-fix`
2. **Ensure tests pass**: Run `pytest` and verify that all gradient checks and shape assertions pass.
3. **Keep changes focused**: If you're fixing a typo in Chapter 0, keep the PR limited to that change.
4. **Follow the Chapter Structure**:
   - `README.md` (textbook explanation + intuition + derivations + common mistakes + exercises)
   - `notebook.ipynb` (runnable walkthrough)
   - `exercises.md` & `solutions/`

---

## Code of Conduct

Be welcoming, polite, and encouraging to all learners and contributors regardless of their level of mathematical or programming experience.

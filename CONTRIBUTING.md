# Contributing

Thanks for taking the time to contribute. Here is everything you need to know.

---

## Contributor License Agreement

Before your first pull request can be merged you need to sign the CLA.
It is a short, plain-English document — read it in [CLA.md](CLA.md).

To sign, add this sentence as a comment on your pull request:

> I have read the CLA in CLA.md and agree to its terms for all my contributions to
> this repository, present and future.

You only need to do this once.

---

## Reporting bugs

Open a [GitHub issue](https://github.com/Ujju999/annotation-automation/issues) with:

- Python version and OS
- Steps to reproduce
- What you expected vs what happened
- Relevant log output or stack trace

---

## Submitting a pull request

1. Fork the repo and create a branch from `master`.
2. Make your changes. Keep commits focused — one logical change per commit.
3. Add or update tests in `tests/` if the change affects logic.
4. Run the test suite locally: `.venv/bin/pytest tests/ -v`
5. Open the PR against `master`. In the description:
   - Explain what the change does and why.
   - Include the CLA sign-off sentence (first contribution only).
   - If it fixes an issue, add `Closes #<number>`.

---

## Code style

- Python 3.11+, no type-ignore comments without explanation.
- No unnecessary comments — name things clearly instead.
- No breaking changes to the public env-var interface without a deprecation path.
- Heavy imports (`ultralytics`, `osam`, `skimage`) must remain lazy so tests load fast.

---

## Questions

Open a GitHub Discussion or issue — there is no mailing list or chat.

# Testing

See [Testing in the README](../../README.md#testing) for suite and individual-test commands. Run tests from the repository root with the development environment activated.

## Choose the test level

- **Pytest for submodules.** Test the public interface with small inputs and direct assertions on results or diagnostics. For a validation test, use the smallest input that reaches the rule under test. Avoid repeating the same behavior across cases unless the variants exercise distinct paths.
- **Lit and FileCheck for end-to-end backend and scheduling behavior.** Use a small, readable graph and schedule that exercise the intended lowering. Check the full deterministic dump rather than walking private IR structures. Where possible, also run the compiled result with validation enabled and check its success: the dump shows what was generated, while execution checks that it works.

## Generate or update lit expectations

Use [`scripts/filecheck/gen_filecheck.py`](../../scripts/filecheck/gen_filecheck.py) as the primary way to generate or update lit expectations. Give the test a `RUN:` directive on its first line, with the command that produces the dump before the pipe to `filecheck`. From the repository root:

```bash
python scripts/filecheck/gen_filecheck.py -i tests/filecheck/backends/specific_test.py
lit -v tests/filecheck/backends/specific_test.py
```

The generator records the **full output dump** as `CHECK` and `CHECK-NEXT`
directives. Prefer this over hand-maintained partial checks: regenerating the
expectations is simpler, and reviewers can see the change in the context of the
surrounding output. Keep the test case small so the full dump remains readable.
Review the regenerated output for the intended change and verify that lit passes. Keep generated blank-line directives as written: trailing whitespace in the generated expectations should be ignored, not trimmed. For a whitespace check on other changes, use `git diff --check -- . ':(exclude)tests/filecheck/'`.

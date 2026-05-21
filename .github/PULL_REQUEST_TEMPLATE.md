<!--
  Thank you for sending a pull request. Please fill in this template
  completely. Reviewers will use this to evaluate the change.
-->

## Summary

<!-- One or two sentences describing what this PR does and why. -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Performance improvement (with measurement)
- [ ] Refactor (no behavior change)
- [ ] Documentation only
- [ ] Build / CI / tooling
- [ ] Breaking change (requires sign-off; see CONTRIBUTING.md)

## Checklist

- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md).
- [ ] `cargo fmt --all --check` passes.
- [ ] `cargo clippy --workspace --all-targets` introduces no new warnings.
- [ ] `cargo test --release --workspace` is green locally on my platform.
- [ ] Tests have been added or updated for the changed behavior.
- [ ] Public API changes (if any) include doc comments and a CHANGELOG entry.
- [ ] Numerical changes (if any) are verified against CRAN R 4.5.3 to 1e-9.
- [ ] No new `unsafe` blocks without justification in the description below.

## Description of changes

<!--
  Walk a reviewer through what changed. Mention:
  - Which crates were touched and why
  - Any architectural decisions worth flagging
  - Any TODOs or follow-ups you're deliberately not addressing here
-->

## Measurements (if perf-related)

<!--
  Required for `perf` PRs. Paste before/after numbers from
  `cargo bench` or `bench/r_vs_r2/`. Hardware specs help.
-->

## Related issues

<!-- Closes #123, related to #456, etc. -->

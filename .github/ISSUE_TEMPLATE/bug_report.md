---
name: Bug report
about: Report something that does not behave the way Ardon-R2 documentation or R 4.5.3 says it should.
title: "[bug] <one-line summary>"
labels: bug
assignees: ''
---

## Describe the bug

<!-- A clear and concise description of what is wrong. -->

## Minimal reproducing code

```r
# Paste the smallest possible Ardon-R2 script that triggers the bug.
# If the bug only shows up on large data, please trim aggressively
# before pasting — even a 10-line repro is dramatically faster to diagnose.
```

## Expected output

<!--
  What did you expect to see?
  If this is a numerical-correctness bug, what does R 4.5.3 produce
  for the same input? Paste R's output if you have it.
-->

## Actual output

<!-- Paste exactly what Ardon-R2 printed, including any error messages. -->

## Environment

- Ardon-R2 version: <!-- run `version()` in the REPL -->
- OS and version: <!-- e.g. Windows 11, Ubuntu 24.04, macOS 14.5 -->
- Architecture: <!-- x86_64, aarch64, etc. -->
- Built from source / pre-built binary:
- Rust toolchain version (if built from source): <!-- `rustc --version` -->

## Severity

- [ ] Numerical correctness — output differs from CRAN R for the same input
- [ ] Crash — Ardon-R2 panicked or segfaulted
- [ ] Wrong behavior — produces no error but does the wrong thing
- [ ] Performance — much slower than expected
- [ ] Cosmetic — formatting, error message wording, etc.

## Additional context

<!--
  Anything else that might help. Recent changes to your script,
  workarounds you tried, related issues you've seen, etc.
-->

# Blender add-on, archived

Taken from commit `dfb7706c2c3d6fa4d634a01b7bc2008aca8f658a` on 2026-09-09.

This is the planning screen as it stood before the standalone application port.
It requires Blender 4.4+ and implements the full feature set listed in section 9
of `docs/superpowers/specs/2026-09-09-standalone-application-design.md`.

## Why it is kept

Two reasons. It is the parity reference the port is diffed against: its scene
output, node poses and mesh volumes are what the headless engine must reproduce.
And it may be revived, because a Blender front end onto the same engine remains a
reasonable thing to want.

## Difference from `legacy/`

`legacy/` holds the frozen script behind the first paper. It is cited, never run,
and never modified. This directory holds working code that is run by the parity
tests and may be brought back.

## Do not edit

Changes belong in `tka_planner/`. Editing this copy destroys its value as a
reference.

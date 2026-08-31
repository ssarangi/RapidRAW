# Project notes for Claude

## Lesson: TS error counts are not a diff

During the "Continue Session" crash investigation, a `ReferenceError: Can't find
variable: useSettingsStore` in `src/hooks/useAppNavigation.ts` caused Continue
Session to fail. The actual bug was trivial: `useSettingsStore` was used via
`.getState()` in three places but never imported.

It took three wrong theories (stale `activeLibraryDbPath`, missing
`active_catalog_root_id` field, a settings-persistence race) and three rounds of
"still broken" from the user before the real cause was found — and only because
the user was asked directly for the console error text, which named the exact
file, line, and undefined variable.

The reason it took so long: throughout the session, `npx tsc --noEmit -p . | wc -l`
was used as a "no new errors introduced" check, comparing the total count before
and after edits (e.g. "152 → 149, looks fine"). That count was never actually
opened and read. The missing-import error was `TS2304: Cannot find name
'useSettingsStore'` and was sitting in that baseline the entire time, plainly
visible, just never inspected.

**Rule going forward:**
- Never treat a tsc/build error *count* as a stand-in for a diff. If the count
  matters, capture the actual list before and after and diff the lists, not
  just the totals — an unchanged count can still hide a severe, unrelated,
  crash-causing error.
- When a user reports a runtime crash (not a logic/UX bug), ask for the actual
  error message/stack trace from devtools/console *before* proposing a fix.
  `ReferenceError: Can't find variable: X` almost always means a missing
  import or a variable used before declaration — check that first, not last,
  and don't reach for state-race or persistence-timing theories until a
  missing-reference explanation has been ruled out by looking at the code.
- If a second attempted fix doesn't resolve a reported crash, stop theorizing
  and get the real error text instead of proposing a third theory.

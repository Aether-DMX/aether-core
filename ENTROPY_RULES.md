# AETHER ENTROPY & DELETION RULES

**Version:** 5.0.0
**Last Updated:** 2026-01-26
**Authority:** Entropy is guaranteed. Deletion is mandatory.

---

## 1. CORE PRINCIPLE

**Code not actively used becomes wrong.**

Every line of code has a maintenance cost. Dead code:
- Confuses future developers
- Breaks during refactors
- Adds cognitive load
- May be accidentally activated

**The only safe dead code is deleted code.**

---

## 2. CODE DELETION RULES

### Rule 1: Unused Code Must Die

| Condition | Action | Timeline |
|-----------|--------|----------|
| Import never used | Delete import | Immediate |
| Function never called | Delete function | Immediate |
| Class never instantiated | Delete class | Immediate |
| File never imported | Delete file | Immediate |
| Branch never taken | Delete branch | After review |

### Rule 2: Commented Code Must Die

| Condition | Action | Exception |
|-----------|--------|-----------|
| Commented-out code | Delete it | None |
| "TODO: uncomment" | Delete it | None |
| "Disabled for now" | Delete it | None |

Git preserves history. Comments are not version control.

### Rule 3: Deprecated Code Has a Deadline

| When Deprecated | Deletion Deadline |
|-----------------|-------------------|
| This release | Next release |
| Last release | This release |
| Two releases ago | OVERDUE - delete now |

### Rule 4: Backup Files Must Die

| Pattern | Action |
|---------|--------|
| `*.bak` | Delete |
| `*.old` | Delete |
| `*.deployed` | Delete |
| `*_backup.*` | Delete |
| `*.orig` | Delete |

Git is the backup system.

---

## 3. TEST DELETION RULES

### Rule 1: Tests Must Test Reality

| Condition | Action |
|-----------|--------|
| Test for deleted code | Delete test |
| Test for removed feature | Delete test |
| Test that always passes | Review or delete |
| Test that always skips | Fix or delete |

### Rule 2: Flaky Tests Must Be Fixed or Deleted

| Flakiness | Action | Deadline |
|-----------|--------|----------|
| Intermittent failure | Fix root cause | 2 weeks |
| Environment-dependent | Make deterministic | 2 weeks |
| Timing-dependent | Add proper waits | 2 weeks |
| Still flaky after fix | DELETE | Immediate |

### Rule 3: Slow Tests Must Justify Existence

| Test Duration | Requirement |
|---------------|-------------|
| < 1 second | OK |
| 1-10 seconds | Must be integration test |
| > 10 seconds | Must be explicitly justified |
| > 60 seconds | Delete or move to separate suite |

---

## 4. FEATURE DELETION RULES

### Rule 1: Unused Features Have Expiry Dates

| Feature State | Action |
|---------------|--------|
| Shipped, never used | Mark for review |
| No usage for 3 months | Candidate for deletion |
| No usage for 6 months | Delete |

### Rule 2: Experimental Features Must Graduate or Die

| Timeline | Action |
|----------|--------|
| Marked experimental | 90-day review deadline |
| Still experimental at 90 days | Decide: promote or delete |
| No decision | Delete |

### Rule 3: Feature Flags Have Lifetimes

| Flag Type | Lifetime |
|-----------|----------|
| Release flag | Until next release |
| Experiment flag | 90 days max |
| Ops flag | Permanent (documented) |

After lifetime expires, remove flag and either:
- Make feature permanent
- Delete feature entirely

---

## 5. CONFIG CLEANUP RULES

### Rule 1: Unused Config Must Die

| Condition | Action |
|-----------|--------|
| Config key never read | Delete it |
| Default always used | Remove override |
| Feature flag always on | Remove flag, keep feature |
| Feature flag always off | Remove flag and feature |

### Rule 2: Environment Variables Must Be Documented

| State | Action |
|-------|--------|
| Undocumented env var | Document or delete |
| Env var with no default | Add default |
| Conflicting env vars | Consolidate |

### Rule 3: Secrets Must Rotate

| Type | Rotation |
|------|----------|
| API keys | Document rotation procedure |
| Passwords | Change on personnel change |
| Tokens | Auto-rotate if possible |

---

## 6. DOCUMENTATION DELETION RULES

### Rule 1: Wrong Docs Are Worse Than No Docs

| Condition | Action |
|-----------|--------|
| Docs describe deleted feature | Delete docs |
| Docs show wrong API | Fix or delete |
| Docs contradict code | Fix docs to match code |

### Rule 2: Stale Docs Must Update or Die

| Staleness | Action |
|-----------|--------|
| Last updated > 1 year | Review for accuracy |
| Last updated > 2 years | Assume wrong, verify |
| Describes removed version | Delete |

### Rule 3: README Files Must Be Accurate

| README State | Action |
|--------------|--------|
| Accurate | Keep |
| Partially accurate | Fix |
| Mostly wrong | Delete (better than lies) |

---

## 7. DEPENDENCY DELETION RULES

### Rule 1: Unused Dependencies Must Be Removed

| Condition | Action |
|-----------|--------|
| Package never imported | Remove from requirements |
| Package only in dev | Move to dev requirements |
| Package with CVE | Update or remove |

### Rule 2: Pin Versions Aggressively

| Type | Pinning |
|------|---------|
| Direct dependency | Pin exact version |
| Transitive dependency | Lock file |
| System package | Document version |

### Rule 3: Audit Dependencies Quarterly

| Check | Action |
|-------|--------|
| Security advisories | Update affected packages |
| Major version drift | Plan upgrade |
| Unmaintained package | Find replacement |

---

## 8. ENFORCEMENT

### Automated Checks

| Check | Tool | Frequency |
|-------|------|-----------|
| Unused imports | linter | Every commit |
| Dead code | coverage + grep | Weekly |
| Unused dependencies | pipreqs | Monthly |

### Manual Reviews

| Review | Frequency | Owner |
|--------|-----------|-------|
| Feature usage | Quarterly | Product |
| Code coverage | Monthly | Engineering |
| Doc accuracy | Per release | Engineering |

### Deletion Ceremony

When deleting significant code:

1. Announce deletion in commit message
2. Reference task/issue number
3. Note what replaces it (if anything)
4. Run tests to verify nothing breaks

---

## 9. EXCEPTIONS

### Valid Reasons to Keep "Dead" Code

| Reason | Requirement |
|--------|-------------|
| Regulatory requirement | Document the regulation |
| Contractual obligation | Document the contract |
| Active development | Branch, don't comment |
| Rollback safety | Time-limited (1 release) |

### Invalid Reasons

| Reason | Response |
|--------|----------|
| "Might need it later" | Git has history |
| "Someone might use it" | They haven't yet |
| "It doesn't hurt" | It does |
| "I don't understand it" | Understand then delete |

---

## 10. PHASE 3 DELETIONS (Reference)

The following were deleted in Phase 3 as examples of proper deletion:

| File | Reason | Replacement |
|------|--------|-------------|
| `playback_controller.py` | Superseded | `unified_playback.py` |
| `DMXService.js` | Authority violation | AETHER Core SSOT |
| `PersistentDMXService.js` | Orphaned | None needed |
| `dmxController.js` | Dead code | None needed |

These deletions followed the rules:
- Code was traced to verify unused
- Replacement was identified (or none needed)
- Tests were updated
- Commit referenced task IDs

---

## 11. DOCUMENT MAINTENANCE

Review this document:
- When deletion rules cause confusion
- When exceptions are granted repeatedly
- When new patterns emerge

**Entropy is certain. Fight it actively or lose.**

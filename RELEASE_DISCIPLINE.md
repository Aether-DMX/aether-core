# AETHER RELEASE & UPGRADE DISCIPLINE

**Version:** 5.0.0
**Last Updated:** 2026-01-26
**Authority:** Defines how AETHER is versioned, released, and upgraded.

---

## 1. VERSIONING STRATEGY

### Semantic Versioning

AETHER uses **Semantic Versioning 2.0.0**: `MAJOR.MINOR.PATCH`

| Component | Meaning | Example |
|-----------|---------|---------|
| MAJOR | Breaking changes | 5.0.0 → 6.0.0 |
| MINOR | New features, backward compatible | 5.0.0 → 5.1.0 |
| PATCH | Bug fixes, backward compatible | 5.0.0 → 5.0.1 |

### What Counts as Breaking

| Change Type | Breaking? | Version Bump |
|-------------|-----------|--------------|
| API endpoint removed | YES | MAJOR |
| API response format changed | YES | MAJOR |
| Database schema incompatible | YES | MAJOR |
| New API endpoint | NO | MINOR |
| New optional parameter | NO | MINOR |
| Bug fix | NO | PATCH |
| Performance improvement | NO | PATCH |
| Documentation update | NO | PATCH |

### Version Locations

Version must be updated in ALL of these:

| Location | File |
|----------|------|
| Backend constant | `aether-core.py:AETHER_VERSION` |
| Package metadata | `pyproject.toml` or `setup.py` |
| API response | `/api/version` |
| Documentation | `SYSTEM_TRUTH.md`, `PRODUCT_TRUTH.md` |

---

## 2. RELEASE PROCESS

### Pre-Release Checklist

Before any release:

- [ ] All tests pass locally
- [ ] All tests pass on Pi
- [ ] Version number updated in all locations
- [ ] CHANGELOG updated
- [ ] TASK_LEDGER.md current
- [ ] No unresolved Critical tasks
- [ ] Documentation reflects code

### Release Types

| Type | Cadence | Testing |
|------|---------|---------|
| PATCH | As needed | Automated + quick manual |
| MINOR | Monthly max | Full regression |
| MAJOR | Rare | Full regression + upgrade testing |

### Release Steps

```bash
# 1. Ensure clean state
git status  # Should be clean

# 2. Run all tests
python -m pytest tests/ -v

# 3. Update version
# Edit AETHER_VERSION in aether-core.py

# 4. Update CHANGELOG
# Document changes since last release

# 5. Commit version bump
git add -A
git commit -m "release: v5.0.1"

# 6. Tag release
git tag -a v5.0.1 -m "Version 5.0.1"

# 7. Push
git push origin main --tags
```

---

## 3. BREAKING CHANGE POLICY

### Definition

A breaking change is any change that:
- Requires user action to maintain functionality
- Changes existing behavior unexpectedly
- Removes or renames public API
- Changes database schema incompatibly

### Requirements for Breaking Changes

| Requirement | Details |
|-------------|---------|
| Justification | Document why change is necessary |
| Migration path | Provide upgrade instructions |
| Deprecation period | One minor version minimum |
| Announcement | Clear communication before release |

### Breaking Change Example

```markdown
## BREAKING CHANGE: v6.0.0

### Change
`/api/looks/{id}/play` now requires `universes` parameter.

### Reason
Explicit universe specification improves reliability.

### Migration
Before:
  POST /api/looks/abc/play

After:
  POST /api/looks/abc/play
  {"universes": [1, 2]}

### Timeline
- v5.2.0: Parameter optional, defaults to [1]
- v5.3.0: Deprecation warning if not specified
- v6.0.0: Parameter required
```

---

## 4. UPGRADE EXPECTATIONS

### For Patch Upgrades (5.0.0 → 5.0.1)

| Expectation | Guarantee |
|-------------|-----------|
| Database compatible | YES |
| No config changes | YES |
| No restart required | NO - restart recommended |
| No data loss | YES |

### For Minor Upgrades (5.0.x → 5.1.0)

| Expectation | Guarantee |
|-------------|-----------|
| Database compatible | YES |
| Config may have new options | Document new options |
| Restart required | YES |
| No data loss | YES |

### For Major Upgrades (5.x → 6.0)

| Expectation | Guarantee |
|-------------|-----------|
| Database migration may be required | Provide migration script |
| Config changes likely | Document all changes |
| Restart required | YES |
| Backup recommended | STRONGLY |

---

## 5. ROLLBACK EXPECTATIONS

### Rollback Capability

| Upgrade Type | Rollback Supported |
|--------------|-------------------|
| PATCH → PATCH | YES |
| MINOR → MINOR | YES |
| MAJOR → MAJOR | LIMITED |

### How to Rollback

```bash
# 1. Stop service
sudo systemctl stop aether-core

# 2. Checkout previous version
git checkout v5.0.0

# 3. Restore database if needed
cp /backup/aether-pre-upgrade.db /srv/aether/core/aether.db

# 4. Restart service
sudo systemctl start aether-core
```

### Rollback Limitations

| Scenario | Can Rollback? |
|----------|---------------|
| Schema unchanged | YES |
| Schema changed, no data loss | YES with migration |
| Schema changed, data added | Data may be lost |
| Schema changed, data modified | NO |

---

## 6. PERMANENT INSTALL SUPPORT

### Commitment

AETHER supports permanent installations:
- Updates are optional, not forced
- Old versions continue to work
- No telemetry or phone-home
- No cloud dependency

### Installer Expectations

| Expectation | Guarantee |
|-------------|-----------|
| Version works indefinitely | YES |
| Updates are stable | Testing required |
| Rollback is possible | YES |
| No forced updates | YES |

### Long-Term Support

| Version | Support Level |
|---------|---------------|
| Current (5.x) | Full support |
| Previous (4.x) | Security fixes only |
| Older | No support |

---

## 7. REMOTE UPDATE SUPPORT

### Supported Update Methods

| Method | Supported |
|--------|-----------|
| Git pull | YES |
| Package manager | Future consideration |
| OTA (nodes) | Separate firmware process |

### Remote Update Process

```bash
# SSH to installation
ssh operator@venue.local

# Stop service
sudo systemctl stop aether-core

# Backup database
cp /srv/aether/core/aether.db /srv/aether/backups/pre-update.db

# Pull update
cd /srv/aether/core
git fetch
git checkout v5.0.1

# Test
python -c "import aether-core"

# Restart
sudo systemctl start aether-core

# Verify
curl http://localhost:8891/api/health
```

---

## 8. INSTALLER TRUST

### Trust Principles

| Principle | Implementation |
|-----------|----------------|
| Reproducible builds | Same commit = same behavior |
| Auditable code | Open source, no obfuscation |
| No hidden behavior | No telemetry, no analytics |
| Documented changes | CHANGELOG for every release |

### Verification

Installers can verify:
- Version matches expected: `/api/version`
- Git commit matches: `git rev-parse HEAD`
- No unauthorized changes: `git status`

---

## 9. CHANGELOG FORMAT

Every release must update CHANGELOG.md:

```markdown
## [5.0.1] - 2026-01-27

### Fixed
- Node ping timeout increased to 5 seconds

### Changed
- Trust heartbeat interval now configurable

### Added
- New /api/trust/events endpoint

### Deprecated
- /api/legacy/endpoint (remove in 6.0.0)

### Removed
- Nothing

### Security
- Updated dependency X for CVE-XXXX
```

---

## 10. EMERGENCY RELEASE PROCESS

For critical security or stability issues:

### Criteria for Emergency Release

| Severity | Examples | Timeline |
|----------|----------|----------|
| CRITICAL | Security vulnerability, data loss | Hours |
| HIGH | System crash, major feature broken | Days |
| MEDIUM | Significant bug | Normal release |

### Emergency Process

1. **Identify** - Confirm severity
2. **Fix** - Minimal change only
3. **Test** - Affected functionality only
4. **Release** - Increment PATCH
5. **Communicate** - Notify affected users
6. **Document** - Update CHANGELOG

---

## 11. DOCUMENT MAINTENANCE

Review this document when:
- Release process changes
- New deployment method added
- Breaking change policy updated
- Support policy changes

**Discipline enables trust. Trust enables adoption.**

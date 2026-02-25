#!/bin/bash
# Phase 2b Deployment — Swap aether-core.py with the slimmed version
#
# Run from: ~/aether-core/
# Usage: bash deploy_slim.sh
#
# This script:
# 1. Backs up the original aether-core.py
# 2. Swaps in aether-core-slim.py
# 3. Runs a compile check
# 4. Restarts the service (if on systemd)
#
# To rollback: cp aether-core.py.pre-slim aether-core.py && sudo systemctl restart aether-core

set -e

echo "═══════════════════════════════════════════"
echo "  Phase 2b: Deploy Slim aether-core.py"
echo "═══════════════════════════════════════════"

# Verify files exist
if [ ! -f "aether-core.py" ]; then
    echo "❌ aether-core.py not found. Run from the aether-core/ directory."
    exit 1
fi

if [ ! -f "aether-core-slim.py" ]; then
    echo "❌ aether-core-slim.py not found. Run migrate_phase2b.py first."
    exit 1
fi

# Check extracted modules exist
MODULES=(
    "core_registry.py"
    "dmx_state_manager.py"
    "playback_state.py"
    "chase_engine_module.py"
    "arbitration_manager.py"
    "schedulers.py"
    "show_engine_module.py"
    "node_manager_module.py"
    "rdm_manager_module.py"
    "content_manager_module.py"
)

echo ""
echo "Checking extracted modules..."
ALL_OK=true
for mod in "${MODULES[@]}"; do
    if [ -f "$mod" ]; then
        echo "  ✓ $mod"
    else
        echo "  ❌ $mod MISSING"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "❌ Missing modules. Cannot deploy."
    exit 1
fi

# Compile check
echo ""
echo "Running compile check on aether-core-slim.py..."
python3 -c "import py_compile; py_compile.compile('aether-core-slim.py', doraise=True)"
echo "  ✓ Compile OK"

# Backup and swap
echo ""
echo "Backing up original (→ aether-core.py.pre-slim)..."
cp aether-core.py aether-core.py.pre-slim
echo "  ✓ Backup saved"

echo "Swapping in slim version..."
cp aether-core-slim.py aether-core.py
echo "  ✓ Swap complete"

# Line count comparison
ORIG_LINES=$(wc -l < aether-core.py.pre-slim)
NEW_LINES=$(wc -l < aether-core.py)
echo ""
echo "  Original: ${ORIG_LINES} lines"
echo "  New:      ${NEW_LINES} lines"
echo "  Reduced:  $((ORIG_LINES - NEW_LINES)) lines removed"

# Check if systemd service exists
if systemctl is-active --quiet aether-core 2>/dev/null; then
    echo ""
    read -p "Restart aether-core service? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo systemctl restart aether-core
        echo "  ✓ Service restarted"
        sleep 2
        if systemctl is-active --quiet aether-core; then
            echo "  ✓ Service is running"
        else
            echo "  ❌ Service failed! Rolling back..."
            cp aether-core.py.pre-slim aether-core.py
            sudo systemctl restart aether-core
            echo "  ✓ Rolled back to original"
            exit 1
        fi
    else
        echo "  Skipped. Restart manually: sudo systemctl restart aether-core"
    fi
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ Deployment complete!"
echo "  Rollback: cp aether-core.py.pre-slim aether-core.py"
echo "═══════════════════════════════════════════"

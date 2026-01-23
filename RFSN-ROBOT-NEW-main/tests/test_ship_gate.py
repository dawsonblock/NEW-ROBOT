"""
Test that enforces the production release checklist.

This test MUST pass before shipping. It parses RELEASE_CHECKLIST.md
and fails if any items are unchecked.
"""

import re
from pathlib import Path
import pytest


def test_release_checklist_all_checked():
    """
    Enforce that all production safety checklist items are checked.

    This test satisfies the meta-requirement: "If any item is unchecked → not shippable."
    """
    checklist_path = Path(__file__).parent.parent / "RELEASE_CHECKLIST.md"

    if not checklist_path.exists():
        pytest.fail(f"RELEASE_CHECKLIST.md not found at {checklist_path}")

    content = checklist_path.read_text(encoding='utf-8')

    # Find all checklist items
    unchecked = []
    checked = []

    for line in content.splitlines():
        # Match: - [ ] item or - [x] item
        match = re.match(r'^\s*-\s*\[([x ])\]\s+(.+)$', line)
        if match:
            is_checked = match.group(1).lower() == 'x'
            item_text = match.group(2).strip()

            if is_checked:
                checked.append(item_text)
            else:
                unchecked.append(item_text)

    # Build failure message
    if unchecked:
        msg = "\n\n" + "="*70 + "\n"
        msg += "PRODUCTION SHIP GATE FAILURE\n"
        msg += "="*70 + "\n"
        msg += "The following checklist items are UNCHECKED:\n\n"
        for item in unchecked:
            msg += f"  ✗ {item}\n"
        msg += "\n"
        msg += f"Checked: {len(checked)}/{len(checked) + len(unchecked)}\n"
        msg += "\nThis system is NOT SHIPPABLE until all items are checked.\n"
        msg += "="*70 + "\n"

        pytest.fail(msg)

    # Success message
    print(f"\n✓ All {len(checked)} production checklist items verified")


def test_ship_gate_module_exists():
    """Verify that the ship_gate module exists and is importable."""
    try:
        from rfsn import ship_gate
        assert hasattr(ship_gate, 'assert_ship_ready')
        assert hasattr(ship_gate, 'seed_everything')
        assert hasattr(ship_gate, 'validate_runtime_safety')
    except ImportError as e:
        pytest.fail(f"ship_gate module not found or not importable: {e}")


def test_ship_gate_config_has_all_requirements():
    """Verify ShipGateConfig covers all checklist items."""
    from rfsn.ship_gate import ShipGateConfig

    config = ShipGateConfig()

    # These attributes must exist and default to True
    assert hasattr(config, 'require_self_collision')
    assert hasattr(config, 'require_safety_manager')
    assert hasattr(config, 'require_learner_firewall')
    assert hasattr(config, 'require_fault_state')
    assert hasattr(config, 'require_orientation_ik')

    # All should default to True (strict mode)
    assert config.require_self_collision is True
    assert config.require_safety_manager is True
    assert config.require_learner_firewall is True
    assert config.require_fault_state is True
    assert config.require_orientation_ik is True


if __name__ == "__main__":
    # Allow running directly
    test_release_checklist_all_checked()
    test_ship_gate_module_exists()
    test_ship_gate_config_has_all_requirements()
    print("\n✓✓✓ All ship gate tests passed ✓✓✓")

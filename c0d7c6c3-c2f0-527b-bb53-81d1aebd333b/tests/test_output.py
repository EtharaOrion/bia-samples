# GENERATED SECTION. DO NOT HAND-EDIT.
# Compiled from solution/rubrics.json. Carries the implied relation only,
# never criterion prose and never reference text.
import json, os, pathlib

REWARD = pathlib.Path(os.environ.get('BIA_REWARD_PATH',
                                    '/logs/verifier/reward.txt'))
RECORD = pathlib.Path(os.environ.get('BIA_REWARD_RECORD',
                                    '/logs/verifier/record.json'))


def _payload():
    return json.loads(RECORD.read_text())


def _outcome(name):
    return _payload()['checkers'].get(name)


def test_reward_file_written():
    assert REWARD.is_file()


def test_bound_reward_is_one_float_agreeing_with_the_record():
    value = float(REWARD.read_text().strip())
    assert 0.0 <= value <= 1.0
    assert abs(value - float(_payload()['reward'])) < 1e-6


def test_reward_shape():
    p = _payload()
    assert 0.0 <= p['reward'] <= 1.0
    assert p['pass'] in (0, 1)
    assert isinstance(p['reason'], str) and p['reason']


def test_zero_reward_carries_reason():
    p = _payload()
    if p['reward'] == 0.0:
        assert p['reason'] != ''


def test_unmeasured_anchors_never_pass():
    p = _payload()
    if p['reason'] == 'anchors-unmeasured':
        assert p['pass'] == 0 and p['reward'] == 0.0


def test_submission_present():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('submission_present') is True


def test_submission_parses():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('submission_parses') is True


def test_bundle_hash_bound():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('bundle_hash_bound') is True


def test_anchors_measured():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('anchors_measured') is True


def test_no_stale_algorithm():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('no_stale_algorithm') is True


def test_no_hardcoded_target():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('no_hardcoded_target') is True


def test_submission_env_isolated():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('submission_env_isolated') is True


def test_run_completes():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('run_completes') is True


def test_checkpoints_present():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('checkpoints_present') is True


def test_frozen_batch_size():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('frozen_batch_size') is True


def test_one_forward_backward_per_step():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('one_forward_backward_per_step') is True


def test_frozen_axes_unchanged():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('frozen_axes_unchanged') is True


def test_uniform_stopping():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('uniform_stopping') is True


def test_seed_interface_present():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('seed_interface_present') is True


def test_no_verifier_edit():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('no_verifier_edit') is True


def test_no_val_shard_touch():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('no_val_shard_touch') is True


def test_target_reached_with_significance():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('target_reached_with_significance') is True


def test_claim_reconciles():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('claim_reconciles') is True


def test_improved_enough_to_pass():
    p = _payload()
    if p['reward'] > 0.0:
        assert _outcome('improved_enough_to_pass') is True


"""An explicit runner batch transition preserves data identity, recomputing coverage."""

import copy

import pytest
from test_sharded_audio import FakeS3, make_index, module


@pytest.mark.parametrize("rank", [0, 1])
def test_changed_batch_recomputes_aligned_validation_without_mutating_source_state(
    tmp_path, monkeypatch, rank
):
    store = FakeS3()
    index = make_index(store, heldout_shards=4)
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    old, _ = module(
        old_dir, monkeypatch, store=store, index=index, rank=rank, world=2, batch_size=2
    )
    source_state = old.state_dict()
    original = copy.deepcopy(source_state)
    changed, _ = module(
        new_dir,
        monkeypatch,
        store=store,
        index=index,
        rank=rank,
        world=2,
        batch_size=4,
        allow_stream_restart_on_resume=True,
        shard_prefetch=1,
    )
    changed.load_state_dict(source_state)
    actual = changed.state_dict()
    assert source_state == original
    for field in (
        "index_sha256",
        "configured_seed",
        "data_mode",
        "naturalistic_sha256",
        "masked_crops_per_recording",
        "masked_crop_samples",
    ):
        assert actual[field] == original[field]
    assert actual["stream_seed"] == original["stream_seed"] + 1
    assert original["evaluation_coverage"]["val"] == {"available": 12, "used": 12, "omitted": 0}
    assert actual["evaluation_coverage"]["val"] == {"available": 12, "used": 8, "omitted": 4}
    batches = list(changed.val_dataloader())
    assert len(batches) == 1 and len(batches[0]["waveform"]) == 4
    # The checkpoint retains its historical cohort coverage; the new publication
    # records current coverage. Runner-owned batch-transition provenance is what
    # authorizes this change, not datamodule state rewriting.

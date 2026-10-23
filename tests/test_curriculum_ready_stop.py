from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.curriculum_ready_stop import NatReadyStop


def callback():
    return NatReadyStop("https://storage.example.com", "audio", "nat/ready.json")


@pytest.mark.parametrize("rank_zero", [True, False])
def test_all_ranks_stop_only_on_broadcast_ready(rank_zero):
    cb = callback()
    cb._is_ready = Mock(return_value=True)
    broadcast = Mock(return_value=True)
    trainer = SimpleNamespace(
        global_step=100,
        is_global_zero=rank_zero,
        strategy=SimpleNamespace(broadcast=broadcast),
        should_stop=False,
    )
    cb.on_train_batch_end(trainer, None, None, None, 0)
    assert trainer.should_stop
    assert cb._is_ready.call_count == int(rank_zero)
    broadcast.assert_called_once_with(rank_zero, src=0)
    cb.on_train_batch_end(trainer, None, None, None, 1)
    assert broadcast.call_count == 1


def test_missing_or_invalid_receipt_does_not_stop_or_leak(monkeypatch, capsys):
    cb = callback()
    cb.client = Mock()
    cb.client.get_object.side_effect = RuntimeError("private signed URL")
    assert cb._is_ready() is False
    assert "private" not in capsys.readouterr().out


def test_unready_keeps_existing_stop_signal():
    cb = callback()
    cb._is_ready = Mock(return_value=False)
    trainer = SimpleNamespace(
        global_step=100,
        is_global_zero=True,
        strategy=SimpleNamespace(broadcast=lambda ready, src: ready),
        should_stop=True,
    )
    cb.on_train_batch_end(trainer, None, None, None, 0)
    assert trainer.should_stop


def test_partial_receipt_cannot_stop_training(monkeypatch):
    cb = callback()
    cb.client = Mock()
    monkeypatch.setattr(
        "scripts.curriculum_ready_stop._bounded_get",
        lambda *args: b'{"status":"partial","ready_for_training":true}',
    )
    assert cb._is_ready() is False


def test_readiness_timeout_is_shorter_than_collective_timeout():
    with pytest.raises(ValueError, match="30 seconds"):
        NatReadyStop(
            "https://storage.example.com", "audio", "nat/ready.json", metadata_timeout_seconds=180
        )

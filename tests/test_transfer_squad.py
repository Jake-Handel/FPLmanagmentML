"""Tests for _apply_pending_transfers and _fpl_sell_price — the logic that
reconstructs a manager's true current squad (and what it's worth) while the
upcoming gameweek's picks endpoint still 404s."""
import pytest

from app.api.routes.optimizer import _apply_pending_transfers, _fpl_sell_price


def _pick(element):
    return {"element": element}


def _transfer(event, out_el, in_el, out_cost=50, in_cost=50, time="2026-11-25T10:00:00Z"):
    return {
        "event": event,
        "element_out": out_el,
        "element_in": in_el,
        "element_out_cost": out_cost,
        "element_in_cost": in_cost,
        "time": time,
    }


def test_no_pending_transfers_returns_picks_unchanged():
    picks = [_pick(e) for e in (1, 2, 3)]
    transfers = [_transfer(5, 1, 9)]  # already reflected in the picks snapshot
    elements, bank_delta, n_pending = _apply_pending_transfers(
        picks, transfers, picks_gameweek=5, target_gameweek=6
    )
    assert elements == {1, 2, 3}
    assert bank_delta == 0.0
    assert n_pending == 0


def test_pending_transfer_swaps_player_and_adjusts_bank():
    picks = [_pick(e) for e in (1, 2, 3)]
    # Sell player 2 (cost 60 -> 6.0m), buy player 9 (cost 45 -> 4.5m): +1.5m
    transfers = [_transfer(6, 2, 9, out_cost=60, in_cost=45)]
    elements, bank_delta, n_pending = _apply_pending_transfers(
        picks, transfers, picks_gameweek=5, target_gameweek=6
    )
    assert elements == {1, 3, 9}
    assert bank_delta == pytest.approx(1.5)
    assert n_pending == 1


def test_transfers_for_later_gameweek_are_ignored():
    picks = [_pick(e) for e in (1, 2, 3)]
    transfers = [_transfer(7, 1, 9)]  # applies to GW7, not our GW6 target
    elements, bank_delta, n_pending = _apply_pending_transfers(
        picks, transfers, picks_gameweek=5, target_gameweek=6
    )
    assert elements == {1, 2, 3}
    assert bank_delta == 0.0
    assert n_pending == 0


def test_multiple_pending_transfers_apply_in_time_order():
    picks = [_pick(e) for e in (1, 2, 3)]
    # Buy 9 then sell 9 again for 8 within the same window — sequential
    # application must end with 8, not keep a phantom 9.
    transfers = [
        _transfer(6, 2, 9, time="2026-11-25T10:00:00Z"),
        _transfer(6, 9, 8, time="2026-11-25T12:00:00Z"),
    ]
    elements, _, n_pending = _apply_pending_transfers(
        picks, transfers, picks_gameweek=5, target_gameweek=6
    )
    assert elements == {1, 3, 8}
    assert n_pending == 2


def test_transfers_when_picks_are_for_target_gameweek_apply_nothing():
    picks = [_pick(e) for e in (1, 2, 3)]
    # Picks already reflect GW6; a GW6 transfer is inside the snapshot.
    transfers = [_transfer(6, 1, 9)]
    elements, bank_delta, n_pending = _apply_pending_transfers(
        picks, transfers, picks_gameweek=6, target_gameweek=6
    )
    assert elements == {1, 2, 3}
    assert bank_delta == 0.0
    assert n_pending == 0


# --- _fpl_sell_price -------------------------------------------------------

def test_sell_price_steady_rise_pays_half_the_gain():
    # Bought at 5.0, rose steadily to 7.0: rises = 2.0 -> sell = 5.0 + 1.0 = 6.0
    assert _fpl_sell_price(5.0, [5.5, 6.0, 6.5, 7.0], 7.0) == pytest.approx(6.0)


def test_sell_price_no_rise_sells_at_market():
    # Bought at 6.0, dropped to 5.0, no rises -> sell at current price
    assert _fpl_sell_price(6.0, [5.5, 5.0], 5.0) == pytest.approx(5.0)


def test_sell_price_capped_at_now_cost_after_drop():
    # Rose then fell below purchase+rises/2 — never sell above market
    sell = _fpl_sell_price(5.0, [7.0, 5.5], 5.5)
    assert sell <= 5.5


def test_sell_price_floored_to_tenth():
    sell = _fpl_sell_price(5.0, [6.13], 6.2)
    assert sell == round(sell, 1)

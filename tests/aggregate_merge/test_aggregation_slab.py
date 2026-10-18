"""CPU coverage for the fixed pass-2 aggregation workspace."""

import torch

from operators.aggregation_slab import (
    DenseAggregationSlab,
    FixedColumnSlab,
)


def test_packed_sum_count_state_uses_one_fixed_storage():
    specs = [(torch.float64, (2,))]
    slab = DenseAggregationSlab(
        -2, 3, torch.int64, specs, ["sum"], "cpu")

    assert slab.nbytes == DenseAggregationSlab.required_bytes(6, specs)
    storage_lo = slab.storage.data_ptr()
    storage_hi = storage_lo + slab.nbytes
    assert storage_lo <= slab.occupied.data_ptr() < storage_hi
    assert storage_lo <= slab.values[0].data_ptr() < storage_hi

    slab.add(
        torch.tensor([1, -2, 1]),
        [torch.tensor(
            [[2.0, 1.0], [5.0, 1.0], [4.0, 1.0]],
            dtype=torch.float64)])
    slab.add(
        torch.tensor([3, -2, 1]),
        [torch.tensor(
            [[7.0, 1.0], [8.0, 1.0], [10.0, 2.0]],
            dtype=torch.float64)])

    keys, values = slab.finish()
    assert torch.equal(keys, torch.tensor([-2, 1, 3]))
    assert torch.equal(
        values[0],
        torch.tensor(
            [[13.0, 2.0], [16.0, 4.0], [7.0, 1.0]],
            dtype=torch.float64))


def test_feed_chunking_does_not_change_slab_result():
    keys = torch.tensor([4, 2, 4, 7, 2, 4, 7, 3])
    states = torch.tensor(
        [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0],
         [5.0, 1.0], [6.0, 1.0], [7.0, 1.0], [8.0, 1.0]],
        dtype=torch.float64)

    one_tick = DenseAggregationSlab(
        2, 7, torch.int64, [(torch.float64, (2,))], ["sum"], "cpu")
    one_tick.add(keys, [states])

    many_ticks = DenseAggregationSlab(
        2, 7, torch.int64, [(torch.float64, (2,))], ["sum"], "cpu")
    for lo, hi in ((0, 1), (1, 5), (5, 6), (6, 8)):
      many_ticks.add(keys[lo:hi], [states[lo:hi]])

    keys_a, values_a = one_tick.finish()
    keys_b, values_b = many_ticks.finish()
    assert torch.equal(keys_a, keys_b)
    assert torch.equal(values_a[0], values_b[0])


def test_amax_and_amin_identities_do_not_poison_first_value():
    slab = DenseAggregationSlab(
        10, 12, torch.int64,
        [(torch.int64, ()), (torch.int64, ())],
        ["amax", "amin"], "cpu")
    slab.add(
        torch.tensor([10, 10, 12]),
        [torch.tensor([-8, -3, -11]), torch.tensor([8, 3, 11])])

    keys, values = slab.finish()
    assert torch.equal(keys, torch.tensor([10, 12]))
    assert torch.equal(values[0], torch.tensor([-3, -11]))
    assert torch.equal(values[1], torch.tensor([3, 11]))


def test_fixed_column_slab_uses_one_storage_for_wide_columns():
    specs = [
        (torch.int64, ()),
        (torch.uint8, (5,)),
        (torch.float64, (2,)),
    ]
    slab = FixedColumnSlab(7, specs, "cpu")

    assert slab.nbytes == FixedColumnSlab.required_bytes(7, specs)
    assert FixedColumnSlab.row_bytes(specs) == 8 + 5 + 16
    storage_lo = slab.storage.data_ptr()
    storage_hi = storage_lo + slab.nbytes
    for column in slab.columns:
      assert storage_lo <= column.data_ptr() < storage_hi
    assert slab.columns[0].shape == (7,)
    assert slab.columns[1].shape == (7, 5)
    assert slab.columns[2].shape == (7, 2)


def test_fixed_column_slab_appends_ticks_and_reuses_capacity():
    specs = [(torch.int64, ()), (torch.float64, (2,))]
    batch = FixedColumnSlab(5, specs, "cpu")

    batch.append([
        torch.tensor([4, 2]),
        torch.tensor([[1.0, 1.0], [2.0, 1.0]], dtype=torch.float64),
    ])
    batch.append([
        torch.tensor([7, 4, 3]),
        torch.tensor(
            [[3.0, 1.0], [4.0, 1.0], [5.0, 1.0]],
            dtype=torch.float64),
    ])
    assert batch.remaining == 0
    assert not batch.can_append([
        torch.tensor([9]),
        torch.tensor([[6.0, 1.0]], dtype=torch.float64),
    ])
    keys, values = batch.tensors()
    assert torch.equal(keys, torch.tensor([4, 2, 7, 4, 3]))
    assert torch.equal(
        values[:, 0], torch.tensor([1., 2., 3., 4., 5.],
                                   dtype=torch.float64))

    batch.reset()
    assert batch.length == 0
    assert batch.remaining == 5
    batch.append([
        torch.tensor([8]),
        torch.tensor([[9.0, 2.0]], dtype=torch.float64),
    ])
    keys, values = batch.tensors()
    assert torch.equal(keys, torch.tensor([8]))
    assert torch.equal(
        values, torch.tensor([[9.0, 2.0]], dtype=torch.float64))

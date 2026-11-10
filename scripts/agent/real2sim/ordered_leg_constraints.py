"""Canonical EFC row permutation for dense, contact-free leg replay; equations unchanged."""
import warp as wp


@wp.kernel
def permutation(count: wp.array(dtype=wp.int32), kind: wp.array2d(dtype=wp.int32),
                identity: wp.array2d(dtype=wp.int32), order: wp.array2d(dtype=wp.int32)):
    world, row = wp.tid()
    n = count[world]
    if row >= n:
        order[world, row] = row
    else:
        rank = int(0)
        for other in range(n):
            earlier = kind[world, other]<kind[world, row]
            if kind[world, other] == kind[world, row]:
                earlier = identity[world, other]<identity[world, row]
                if identity[world, other] == identity[world, row]: earlier = other<row
            if earlier: rank += 1
        order[world, rank] = row


@wp.kernel
def gather_float(order: wp.array2d(dtype=wp.int32), source: wp.array2d(dtype=wp.float32),
                 target: wp.array2d(dtype=wp.float32)):
    world, row = wp.tid(); target[world, row] = source[world, order[world, row]]


@wp.kernel
def gather_int(order: wp.array2d(dtype=wp.int32), source: wp.array2d(dtype=wp.int32),
               target: wp.array2d(dtype=wp.int32)):
    world, row = wp.tid(); target[world, row] = source[world, order[world, row]]


@wp.kernel
def gather_jacobian(order: wp.array2d(dtype=wp.int32), source: wp.array3d(dtype=wp.float32),
                    target: wp.array3d(dtype=wp.float32)):
    world, row, dof = wp.tid(); target[world, row, dof] = source[world, order[world, row], dof]


def buffers(model, data):
    assert not model.is_sparse and model.nflex == 0
    # 排序仅用于无接触、无岛划分的固定机身诊断；不接入训练后端。
    assert not model.opt.enableflags
    integers = ('type', 'id', 'state', 'island')
    floats = ('pos', 'margin', 'D', 'vel', 'aref', 'frictionloss', 'force', 'Jqvel')
    arrays = {name:wp.empty_like(getattr(data.efc, name)) for name in integers+floats
              if all(getattr(data.efc, name).shape)}
    arrays['J'] = wp.empty_like(data.efc.J)
    rows = max(array.shape[1] for array in arrays.values())
    return wp.empty((data.nworld, rows), dtype=wp.int32), arrays, integers


def apply(data, storage):
    order, arrays, integers = storage
    wp.launch(permutation, dim=order.shape, inputs=[data.nefc, data.efc.type, data.efc.id, order])
    for name, temporary in arrays.items():
        source = getattr(data.efc, name)
        kernel = gather_jacobian if name == 'J' else gather_int if name in integers else gather_float
        wp.launch(kernel, dim=source.shape, inputs=[order, source, temporary])
        wp.copy(source, temporary)

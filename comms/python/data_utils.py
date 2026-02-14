import struct
import torch


# ======= DATA STRUCT HANDLING =======

DTYPE_TO_ID = {
    torch.float16: 1,
    torch.float32: 2,
    torch.int8: 3,
    torch.int16: 4,
    torch.int32: 5,
    torch.uint8: 6,
    torch.uint16: 7,
    torch.uint32: 8,
    torch.int64: 9,
    torch.uint64: 10,
}

ID_TO_DTYPE = {i: dtype for dtype, i in DTYPE_TO_ID.items()}

DTYPE_TO_BYTESIZE = {
    torch.float16: 2,
    torch.float32: 4,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.uint8: 1,
    torch.uint16: 2,
    torch.uint32: 4,
    torch.int64: 8,
    torch.uint64: 8,
}


def unpack_float16(data: bytes) -> float:
    return struct.unpack('e', data)[0]


def unpack_float32(data: bytes) -> float:
    return struct.unpack('f', data)[0]


def unpack_int8(data: bytes) -> int:
    return struct.unpack('b', data)[0]


def unpack_int16(data: bytes) -> int:
    return struct.unpack('h', data)[0]


def unpack_int32(data: bytes) -> int:
    return struct.unpack('i', data)[0]


def unpack_uint8(data: bytes) -> int:
    return struct.unpack('B', data)[0]


def unpack_uint16(data: bytes) -> int:
    return struct.unpack('H', data)[0]


def unpack_uint32(data: bytes) -> int:
    return struct.unpack('I', data)[0]


DTYPE_TO_UNPACK_METHOD = {
    torch.float16: unpack_float16,
    torch.float32: unpack_float32,
    torch.int8: unpack_int8,
    torch.int16: unpack_int16,
    torch.int32: unpack_int32,
    torch.uint8: unpack_uint8,
    torch.uint16: unpack_uint16,
    torch.uint32: unpack_uint32,
}


def pack_float16(f: float) -> bytes:
    return struct.pack('e', f)


def pack_float32(f: float) -> bytes:
    return struct.pack('f', f)


def pack_int8(i: int) -> bytes:
    return struct.pack('b', i)


def pack_int16(i: int) -> bytes:
    return struct.pack('h', i)


def pack_int32(i: int) -> bytes:
    return struct.pack('i', i)


def pack_int64(i: int) -> bytes:
    return struct.pack('q', i)


def pack_uint8(i: int) -> bytes:
    return struct.pack('B', i)


def pack_uint16(i: int) -> bytes:
    return struct.pack('H', i)


def pack_uint32(i: int) -> bytes:
    return struct.pack('I', i)


def pack_uint64(i: int) -> bytes:
    return struct.pack('Q', i)


DTYPE_TO_PACK_METHOD = {
    torch.float16: pack_float16,
    torch.float32: pack_float32,
    torch.int8: pack_int8,
    torch.int16: pack_int16,
    torch.int32: pack_int32,
    torch.int64: pack_int64,
    torch.uint8: pack_uint8,
    torch.uint16: pack_uint16,
    torch.uint32: pack_uint32,
    torch.uint64: pack_uint64,
}


def to_tensor_bytes(tensor: torch.Tensor, dtype=None) -> bytes:
    """
    Converts a tensor into bytes in the following format:

    dtypeID(uint8) +

    ndims(uint8) +

    dims(uint32)[nof=ndims] +

    data(dtype(dtypeID))[nof=product(dims)]

    :param tensor: the tensor to be converted.
    :param dtype: explicitly annotated, dtype of the tensor to be converted.
    :return: the resulting converted bytes.
    """
    if dtype is None:
        dtype = DTYPE_TO_ID[tensor.dtype]
    else:
        dtype = DTYPE_TO_ID[dtype]
    ndims = len(tensor.shape)
    b = DTYPE_TO_PACK_METHOD[torch.uint8](dtype) + DTYPE_TO_PACK_METHOD[torch.uint8](ndims)
    for dim in tensor.shape:
        b += DTYPE_TO_PACK_METHOD[torch.uint32](dim)
    b += tensor.cpu().numpy().tobytes()
    return b


def wrap_sendableIDs(*args) -> bytes:
    d = bytes()
    for i in args:
        d += pack_uint16(i)
    return d


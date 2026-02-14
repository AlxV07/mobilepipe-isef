import json
import socket
import struct
from typing import Optional

import torch
from mobilepipe.comm.data_utils import ID_TO_DTYPE, DTYPE_TO_BYTESIZE, DTYPE_TO_UNPACK_METHOD, DTYPE_TO_PACK_METHOD, to_tensor_bytes

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 12345  # local port `iproxy` maps to default `iPhone:2346`


# ======= COMM HANDING =======

class OperationID:
    weight = 0     # a weight tensor is sent
    input = 1      # a layer input tensor is sent (and output expected)
    bias = 2       # a bias tensor is sent
    parameter = 3


class InputOperationID:
    Linear = 0
    Qwen3_MLP = 1
    Qwen3_RMSNorm = 2
    Qwen3_Attention = 3
    Qwen3_Decoder = 4
    Test = 5
    LayerNorm = 6
    Dropout = 7
    GPT2_Block = 8
    GPT2_Attention = 9
    GPT2_MLP = 10
    Conv1D = 11
    GELU = 12
    GPT2 = 13
    Conv2D = 14
    BatchNorm2D = 15
    ResNet_BasicBlock = 16
    ResNet = 17
    ResNet_Train = 18
    CrossEntropy = 19
    ResNet_Stage2 = 20
    GPT2_Train = 21
    ResNet_BatchInference = 22
    Qwen3_S2Inference = 23
    AgentTool_VectorDBSearch = 24
    MobilePipe_ResNet_Train = 25
    MobilePipe_DistilBERTTransformer_Train = 26
    MobilePipe_DistilBERTRegression_Train = 27


class SendableIDGenerator:
    def __init__(self):
        self.idx = 0

    def generate_ID(self) -> int:
        self.idx += 1
        return self.idx


class CommHandler:
    """
    When creating a SendableLayer, pass a CommHandler object `comm=COMM_HANDLER` in the constructor.
    This will:
    - use `self.sendableID = comm.sendableID_generator.generate_ID()`, assigning an ID to the layer
    - use `self.sendable = comm.is_sendable(self.sendableID)`, marking whether this layer is toggled on

    *Note: SendableQwen3 currently uses a non-standard approach - was written before establishment of this protocol
    TODO: rewrite SendableQwen3 to use standardized framework
    """

    def __init__(self):
        self._s: Optional[socket.socket] = None
        self.HOST = None
        self.PORT = None

        self.sendableID_generator = None
        self.sendables = set()

    def mark_as_sendable(self, i: int):
        self.sendables.add(i)

    def mark_as_not_sendable(self, i: int):
        self.sendables.remove(i)

    def is_sendable(self, i: int):
        return i in self.sendables

    def connect(self, default=False):
        if default:
            self.HOST = DEFAULT_HOST
            self.PORT = DEFAULT_PORT
        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._s.connect((self.HOST, self.PORT))

    def set_host(self, host):
        self.HOST = host

    def set_port(self, port):
        self.PORT = port

    def get_socket(self) -> socket.socket:
        return self._s

    def receive_exactly(self, nof_bytes) -> bytes:
        """
        Receive exactly ```nof_bytes``` bytes from socket.
        :param nof_bytes: number of bytes to receive.
        :return: received bytes.
        """
        assert self._s
        data = bytearray()
        while len(data) < nof_bytes:
            p = self._s.recv(nof_bytes - len(data))
            if not p:
                raise ConnectionError(f'CommsHandler: socket closed after receiving {len(data)}/{nof_bytes} bytes.')
            data.extend(p)
        return bytes(data)

    def receive_tensor(self) -> torch.Tensor:
        """
        Receives a tensor from socket in the following format:

        dtypeID(uint8) +

        ndims(uint8) +

        dims(uint16)[nof=ndims] +

        data(dtype(dtypeID))[nof=product(dims)]

        :return: torch.Tensor from received data.
        """
        dtype = ID_TO_DTYPE[DTYPE_TO_UNPACK_METHOD[torch.uint8](self.receive_exactly(DTYPE_TO_BYTESIZE[torch.uint8]))]
        ndims = DTYPE_TO_UNPACK_METHOD[torch.uint8](self.receive_exactly(DTYPE_TO_BYTESIZE[torch.uint8]))
        dims = []
        total_byte_size = DTYPE_TO_BYTESIZE[dtype]
        for _ in range(ndims):
            dims.append(DTYPE_TO_UNPACK_METHOD[torch.uint16](self.receive_exactly(DTYPE_TO_BYTESIZE[torch.uint16])))
            total_byte_size *= dims[-1]
        return torch.frombuffer(bytearray(self.receive_exactly(total_byte_size)), dtype=dtype).reshape(*dims)

    def receive_double(self) -> float:
        return struct.unpack('!d', self._s.recv(8))[0]

    def send_operation(self, op_id: int, data: bytes):
        """
        Sends operation to socket in following format:

        op_id(uint8) +

        data

        :param op_id: the operation id to send.
        :param data: the operation data to send.
        """
        self._s.sendall(DTYPE_TO_PACK_METHOD[torch.uint8](op_id) + data)

    def send_uint8(self, x: int):
        """
        Sends `x` as uint8 to the socket.
        :param x: the int to send.
        """
        self._s.sendall(DTYPE_TO_PACK_METHOD[torch.uint8](x))

    def send_weight(self, sendableID: int, tensor: torch.Tensor):
        """
        Sends weight operation to socket in following format:

        op_id=weight(uint8) +

        sendable_id(uint16) +

        weight_tensor

        :param sendableID: id to store this weight as.
        :param tensor: the weight tensor to send.
        """
        self.send_operation(OperationID.weight,
                            DTYPE_TO_PACK_METHOD[torch.uint16](sendableID) +
                            to_tensor_bytes(tensor))

    def send_sendableID(self, sendableID: int):
        """
        Sends sendableID to socket in following format:

        sendable_id(uint16)

        :param sendableID: the id to send.
        """
        self._s.sendall(DTYPE_TO_PACK_METHOD[torch.uint16](sendableID))

    def send_tensor(self, tensor: torch.Tensor, dtype=None, print_shape=False, print_func=print):
        """
        Sends tensor to socket in expected format.
        :param dtype: explicitly specify the dtype of tensor to send.
        :param tensor: The tensor to send.
        :param print_shape: debug variable - if True, prints shape of using the given `print_func`.
        :param print_func: the print function to use if `print_tensor_shape` is True.
        """
        self._s.sendall(to_tensor_bytes(tensor, dtype=dtype))
        if print_shape:
            print_func(tensor.shape)

    def send_all_weights(self, tensors: list, startId: int = 1):
        """
        Sends all weights to socket in expected format, assigning sendableIDs incrementally from left to right.
        :param tensors: List of tensors to send.
        :param startId: The first ID to start assigning sendableIDs to tensors from.
        """
        op_id_bytes = DTYPE_TO_PACK_METHOD[torch.uint8](OperationID.weight)
        pack_id = DTYPE_TO_PACK_METHOD[torch.uint16]
        self._s.sendall(b"".join([op_id_bytes + pack_id(i + startId) + to_tensor_bytes(tensors[i]) for i in range(len(tensors))]))

    def send_parameter(self, sendableID: int, tensor: torch.Tensor):
        """
        Sends parameter operation to socket in following format:

        op_id=weight(uint8) +

        sendable_id(uint16) +

        weight_tensor

        :param sendableID: id to store this weight as.
        :param tensor: the weight tensor to send.
        """
        self.send_operation(OperationID.parameter,
                            DTYPE_TO_PACK_METHOD[torch.uint16](sendableID) +
                            to_tensor_bytes(tensor))

    def send_all_parameters(self, tensors: list, startId: int = 1):
        """
        Sends all weights to socket in expected format, assigning sendableIDs incrementally from left to right.
        :param tensors: List of tensors to send.
        :param startId: The first ID to start assigning sendableIDs to tensors from.
        """
        op_id_bytes = DTYPE_TO_PACK_METHOD[torch.uint8](OperationID.parameter)
        pack_id = DTYPE_TO_PACK_METHOD[torch.uint16]
        self._s.sendall(b"".join([op_id_bytes + pack_id(i + startId) + to_tensor_bytes(tensors[i]) for i in range(len(tensors))]))

    def send_input(self, input_operation_id: int, tensor: torch.Tensor = None, misc_data: bytes = None):
        """
        Sends input operation to socket in following format:

        op_id=input(uint8) +

        input_operation(uint16) +

        misc_data (optional) +

        input_tensor

        :param input_operation_id: id to store this weight as.
        :param tensor: the weight tensor to send.  If none, only `input_operation_id` is sent.
        :param misc_data: (optional) extra data to send;
                          handling input_operation_id on client side should recognize and receive misc_data as required
        """
        d = DTYPE_TO_PACK_METHOD[torch.uint16](input_operation_id)
        if tensor is not None:
            d += to_tensor_bytes(tensor)
        if misc_data is not None:
            d += misc_data
        self.send_operation(OperationID.input, d)
        # NOTE: for now, results from operations performed on input should be received using `receive_tensor`

    def send_run_input_op_test(self):
        """
        Sends the test input operation to socket:

        op_id=input(uint8) +

        input_operation=test(uint16)
        """
        self.send_operation(OperationID.input, DTYPE_TO_PACK_METHOD[torch.uint16](InputOperationID.Test))

    def set_sendableID_generator(self, sendableID_generator: SendableIDGenerator):
        self.sendableID_generator = sendableID_generator

    def send_traced(self, traced_model: dict):
        """
        Sends a traced model representation over the socket.

        :param traced_model: The traced model representation from the trace method
        """
        json_bytes = json.dumps(traced_model).encode('utf-8')
        nof_bytes = len(json_bytes)
        self._s.sendall(DTYPE_TO_PACK_METHOD[torch.uint32](nof_bytes))
        self._s.sendall(json_bytes)


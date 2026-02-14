import Foundation
import Network
import MetalPerformanceShaders

enum SendableError : Error {
    case failedIDReception(_ reason: String)
}

class OperationID {
    static let parameter = 3
    static let input = 1
}

class InputOperationID {
    static let MobilePipe_ResNet_Train = 25
    static let MobilePipe_DistilBERTTransformer_Train = 26
    static let MobilePipe_DistilBERTRegression_Train = 27
}

class CommHandler {
    init() {}
    
    let type: NWParameters = NWParameters.tcp
    let port: NWEndpoint.Port = 6789
    var networkListener: NWListener?
    var connection: NWConnection?
    
    func createListener() {
        /*
         Creates & stores a NWListener with this CommHandler's type & port.
         */
        do {
            self.networkListener = try NWListener(using: type, on: port)
        } catch {
            fatalError("Failed to create NWListener: \(error)")
        }
    }
    
    func setConnection(_ connection: NWConnection) {
        self.connection = connection
    }
    
    func setNewConnectionHandler(_ handler: @escaping (NWConnection) -> Void) {
        self.networkListener!.newConnectionHandler = handler
    }
    
    func startListener() {
        self.networkListener!.start(queue: .main)
    }

    func receiveExactly(_ expectedBytes: Int) async throws -> Data {
        /*
         Receive exactly `expectedBytes` bytes.
         :returns: the bytes in a Data object.
         */
        var received = Data()
        while received.count < expectedBytes {
            let bytesNeeded = expectedBytes - received.count
            let chunk: Data = try await withCheckedThrowingContinuation { continuation in
                self.connection!.receive(minimumIncompleteLength: 1, maximumLength: bytesNeeded) { data, _, isComplete, error in
                    if let error = error {
                        continuation.resume(throwing: error)
                        return
                    }
                    if let data = data {
                        continuation.resume(returning: data)
                    } else {
                        continuation.resume(returning: Data())
                    }
                }
            }
            received.append(chunk)
        }
        return received
    }
    
    func receiveUInt8() async throws -> Int {
        return DataHandler.unpackUInt8IntoInt(try await self.receiveExactly(DataHandler.DTYPE_TO_BYTESIZE[.uInt8]!))
    }
    
    func receiveUInt16() async throws -> Int {
        return DataHandler.unpackUInt16IntoInt(try await self.receiveExactly(DataHandler.DTYPE_TO_BYTESIZE[.uInt16]!))
    }

    func receiveOpID() async throws -> Int {
        /*
         Receives the op_id (uint8) of the current operation from the socket.
         */
        do {
            return try await receiveUInt8()
        } catch {
            throw SendableError.failedIDReception("Failed to receive op_id: \(error)")
        }
    }
    
    func receiveSendableID() async throws -> Int {
        /*
         Receives a sendableID (uint16) from the socket.
         */
        do {
            return try await receiveUInt16()
        } catch {
            throw SendableError.failedIDReception("Failed to receive selectableID: \(error)")
        }
    }

    func receiveSendableTensor() async throws -> SendableTensor {
        /*
         Receive a tensor from socket in the expected format:
         
         dtypeID(uint8) +
         ndims(uint8) +
         dims(uint16)[nof=ndims] +
         data(dtype(dtypeID))[nof=product(dims)]
         
         :returns: the tensor wrapped in a SendableTensor object.
         */
        let i = DataHandler.unpackUInt8IntoInt(try await self.receiveExactly(DataHandler.DTYPE_TO_BYTESIZE[.uInt8]!))
        let dataType: MPSDataType = DataHandler.ID_TO_DTYPE[i]!
        let ndims: Int = DataHandler.unpackUInt8IntoInt(try await self.receiveExactly(DataHandler.DTYPE_TO_BYTESIZE[.uInt8]!))
        var dims: [NSNumber] = []
        var nofBytes = DataHandler.DTYPE_TO_BYTESIZE[dataType]!
        for _ in 0..<ndims {
            let dim: Int = DataHandler.unpackUInt32IntoInt(try await self.receiveExactly(DataHandler.DTYPE_TO_BYTESIZE[.uInt32]!))
            dims.append(dim as NSNumber)
            nofBytes *= dim
        }
        let tensorData = try await self.receiveExactly(nofBytes)
        return SendableTensor(dataType: dataType, dims: dims, nofBytes: nofBytes, tensorData: tensorData)
    }
    
    func sendSendableTensor(tensor: SendableTensor) {
        /*
         Sends a tensor to the socket in the expected format:
         
         dtypeID(uint8) +
         ndims(uint8) +
         dims(uint16)[nof=ndims] +
         data(dtype(dtypeID))[nof=product(dims)]
         */
        var data = Data()
        
        var dataTypeId: UInt8 = UInt8(DataHandler.DTYPE_TO_ID[tensor.dataType]!)
        data.append(Data(bytes: &dataTypeId, count: DataHandler.DTYPE_TO_BYTESIZE[.uInt8]!))
        
        var ndims = UInt8(tensor.dims.count)
        data.append(Data(bytes: &ndims, count: DataHandler.DTYPE_TO_BYTESIZE[.uInt8]!))

        for dim in tensor.dims {
            var d = UInt16(truncating: dim)
            data.append(Data(bytes: &d, count: DataHandler.DTYPE_TO_BYTESIZE[.uInt16]!))
        }
        
        data.append(tensor.tensorData)
        
        self.connection!.send(content: data, completion: .contentProcessed { _ in })
    }
    
    func sendDouble(_ t: Double) {
        var number = t.bitPattern.bigEndian  
        let data = Data(bytes: &number, count: MemoryLayout<UInt64>.size)
        self.connection!.send(content: data, completion: .contentProcessed { error in })
    }
}


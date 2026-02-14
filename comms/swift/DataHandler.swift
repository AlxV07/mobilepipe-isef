import MetalPerformanceShaders
import MetalPerformanceShadersGraph


class CounterFrom1 {
    var value: Int
    init() {
        self.value = 0
    }
    func inc() -> Int {
        self.value += 1
        return self.value
    }
}


class DataHandler {
    static let DTYPE_TO_ID: [MPSDataType: Int] = [
        .float16: 1,
        .float32: 2,
        .int8: 3,
        .int16: 4,
        .int32: 5,
        .uInt8: 6,
        .uInt16: 7,
        .uInt32: 8,
        .int64: 9,
        .uInt64: 10,
    ]
    static let ID_TO_DTYPE: [Int: MPSDataType] = [
        1: .float16,
        2: .float32,
        3: .int8,
        4: .int16,
        5: .int32,
        6: .uInt8,
        7: .uInt16,
        8: .uInt32,
        9: .int64,
        10: .uInt64,
    ]
    static let DTYPE_TO_BYTESIZE: [MPSDataType: Int] = [
        .float16: 2,
        .float32: 4,
        .int8: 1,
        .int16: 2,
        .int32: 4,
        .uInt8: 1,
        .uInt16: 2,
        .uInt32: 4,
        .int64: 8,
        .uInt64: 8,
    ]
    
    static func unpackUInt8IntoInt(_ data: Data) -> Int {
        return Int(data.withUnsafeBytes { $0.load(as: UInt8.self) })
    }
    
    static func unpackUInt16IntoInt(_ data: Data) -> Int {
        return Int(data.withUnsafeBytes { $0.load(as: UInt16.self) })
    }
    
    static func unpackUInt32IntoInt(_ data: Data) -> Int {
        return Int(data.withUnsafeBytes { $0.load(as: UInt32.self) })
    }
    
    static func toSendableFloat32Tensor(_ r: MPSGraphTensorData) -> SendableTensor {
        var resSize = 4  // 4 bytes
        for dim in r.shape {
            resSize *= Int(truncating: dim)
        }
        var resBytes = [Float32](repeating: 0, count: resSize)
        let rA = r.mpsndarray()
        rA.readBytes(&resBytes, strideBytes: nil)
        let d = Data(bytes: resBytes, count: resSize)
        let res: SendableTensor = SendableTensor(dataType: .float32, dims: rA.descriptor().getShape(), nofBytes: resSize, tensorData: d)
        return res
    }
    
    static func toSendableFloat16Tensor(_ r: MPSGraphTensorData) -> SendableTensor {
        var resSize = 2  // 2 bytes
        for dim in r.shape {
            resSize *= Int(truncating: dim)
        }
        var resBytes = [Float16](repeating: 0, count: resSize)
        let rA = r.mpsndarray()
        rA.readBytes(&resBytes, strideBytes: nil)
        let d = Data(bytes: resBytes, count: resSize)
        let res: SendableTensor = SendableTensor(dataType: .float16, dims: rA.descriptor().getShape(), nofBytes: resSize, tensorData: d)
        return res
    }
    
    // ======= Debug Printable Tensor Methods =======
    
    static func toSwiftFloat32Array(_ mpsArray: MPSNDArray) -> [Float32] {
        var l = 1
        for i in 0..<mpsArray.numberOfDimensions {
            l *= Int(mpsArray.length(ofDimension: i))
        }
        var result = [Float32](repeating: 0, count: l)
        mpsArray.readBytes(&result, strideBytes: nil)
        return result
    }
    
    static func toSwiftFloat16Array(_ mpsArray: MPSNDArray) -> [Float16] {
        var l = 1
        for i in 0..<mpsArray.numberOfDimensions {
            l *= Int(mpsArray.length(ofDimension: i))
        }
        var result = [Float16](repeating: 0, count: l)
        mpsArray.readBytes(&result, strideBytes: nil)
        return result
    }
    
    static func toSwiftFloat32Array(_ data: Data) -> [Float32] {
        let count = data.count / 4
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float32.self).prefix(count))
        }
    }
}


import Foundation
import MetalPerformanceShaders


class SendableTensor {
    let dataType: MPSDataType
    let dims: [NSNumber]
    let nofBytes: Int
    let tensorData: Data
    
    init(dataType: MPSDataType, dims: [NSNumber], nofBytes: Int, tensorData: Data) {
        self.dataType = dataType
        self.dims = dims
        self.nofBytes = nofBytes
        self.tensorData = tensorData
    }
}


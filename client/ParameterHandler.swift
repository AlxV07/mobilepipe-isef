import MetalPerformanceShadersGraph


class ParameterHandler {
    init() {}
    
    var device: MTLDevice?

    func setDevice(_ device: MTLDevice) {
        self.device = device
    }
    
    var parameterMPSGraphTensorData: [Int : MPSGraphTensorData] = [:]
    var parameterSendables: [Int : SendableTensor] = [:]

    func storeParameter(sendableID: Int, tensor: SendableTensor) {
        guard let weightBuffer = device!.makeBuffer(
            bytes: (tensor.tensorData as NSData).bytes,
            length: tensor.nofBytes,
            options: .storageModeShared)
        else {
            fatalError("Failed to make weight buffer.")
        }
        self.parameterMPSGraphTensorData[sendableID] = MPSGraphTensorData(weightBuffer, shape: tensor.dims, dataType: tensor.dataType)
        self.parameterSendables[sendableID] = tensor
    }
    
    func getParameterTensorData(sendableID: Int) -> MPSGraphTensorData {
        return self.parameterMPSGraphTensorData[sendableID]!
    }
    
    func setParameterTensorData(sendableID: Int, data: MPSGraphTensorData) {
        self.parameterMPSGraphTensorData[sendableID] = data
    }

    func getParameterShape(sendableID: Int) -> [NSNumber] {
        return self.parameterSendables[sendableID]!.dims
    }
    
    func getParameterNofBytes(layerId: Int) -> Int {
        return self.parameterSendables[layerId]!.nofBytes
    }
}


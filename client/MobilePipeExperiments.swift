
import MetalPerformanceShaders
import MetalPerformanceShadersGraph


class MobilePipeExperiments {
    static func MobilePipe_ResNet_Train(_ ENV: IAEnvironment) async throws {
        try await inputOperation_ResNet_DynamicPipeline(ENV)
    }
    
    static func inputOperation_ResNet_DynamicPipeline(_ ENV: IAEnvironment) async throws {
        let NOF_DYNAMIC_TEST_BATCHES = try await ENV.commHandler.receiveUInt8()  // nof batches per dynamic test
        let NOF_MICROBATCHES = try await ENV.commHandler.receiveUInt8()
        let SCALE_LR = try await ENV.commHandler.receiveUInt8()

        var optimizer = try await DynamicPipelineConfigurations.C1(ENV, NOF_DYNAMIC_TEST_BATCHES, NOF_MICROBATCHES, SCALE_LR)
        optimizer = try await DynamicPipelineConfigurations.C2(ENV, NOF_DYNAMIC_TEST_BATCHES, optimizer)
        optimizer = try await DynamicPipelineConfigurations.C3(ENV, NOF_DYNAMIC_TEST_BATCHES, optimizer)

        let bestConfig = try await ENV.commHandler.receiveUInt8()
        switch bestConfig {
            case 1:
                try await ContinuedTraining.C1(ENV, optimizer)
                break
            case 2:
                try await ContinuedTraining.C2(ENV, optimizer)
                break
            case 3:
                try await ContinuedTraining.C3(ENV, optimizer)
                break
            default:
                fatalError("invalid bestConfig")
        }
        // Final syncing is handleded within ContinuedTraining
    }
}


class MobilePipeUtils {
    static let RESNET34_S2_NOF_PARAMS: Int = 147  // all params for: layer 2 + layer3 + layer4 + end
    static let RESNET34_L2_PARAMS = 45
    static let RESNET34_L3_PARAMS = 65

    actor TensorQueue<T> {
        private var queue: [T] = []
        private var waiters: [CheckedContinuation<T, Never>] = []

        func push(_ item: T) {
            if waiters.isEmpty {
                queue.append(item)
            } else {
                let waiter = waiters.removeFirst()
                waiter.resume(returning: item)
            }
        }

        func pop() async -> T {
            if !queue.isEmpty {
                return queue.removeFirst()
            }

            return await withCheckedContinuation { continuation in
                waiters.append(continuation)
            }
        }
    }
    
    static let GlobalTensorQueue = TensorQueue<SendableTensor>()
    
    static func placeHolderFromId(_ ENV: IAEnvironment, _ graph: MPSGraph, _ id: Int) -> MPSGraphTensor {
        return ENV.graphHandler.makePlaceholderFloat32Tensor(graph: graph,
                                                             shape: ENV.parameterHandler.getParameterShape(sendableID: id),
                                                             name: "Tensor\(id)")
    }
    
    static var CachedZeroTensorData: [[NSNumber] : MPSGraphTensorData] = [:]

    static func zerosTensorData(_ ENV: IAEnvironment, shape: [NSNumber]) -> MPSGraphTensorData {
        if MobilePipeUtils.CachedZeroTensorData.keys.contains(shape) {
            return MobilePipeUtils.CachedZeroTensorData[shape]!
        }
        let byteCount = 4 * shape.map { $0.intValue }.reduce(1, *)
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: MemoryLayout<Float>.alignment)
        buffer.initializeMemory(as: UInt8.self, repeating: 0, count: byteCount)
        let p = UnsafeRawPointer(buffer)
        guard let buffer = ENV.parameterHandler.device!.makeBuffer( bytes: p, length: byteCount, options: .storageModeShared)
        else {
            fatalError("Failed to make zero buffer.")
        }
        let d = MPSGraphTensorData(buffer, shape: shape, dataType: .float32)
        MobilePipeUtils.CachedZeroTensorData[shape] = d
        return d
    }
    
    static func paramIdIsTrainable(_ id: Int) -> Bool {
        // if convolution or fc.weight or fc.bias
        return (id - 1) % 5 == 0 || id == MobilePipeUtils.RESNET34_S2_NOF_PARAMS - 1 || id == MobilePipeUtils.RESNET34_S2_NOF_PARAMS
    }
}


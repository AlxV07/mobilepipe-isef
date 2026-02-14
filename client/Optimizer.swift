import MetalPerformanceShadersGraph


class Optimizer {
    var param_ids: [Int] = [];
    var PARAM_ID_TO_IDX: [Int: Int] = [:];
    
    func setup(param_ids: [Int], NOF_MICROBATCHES: Int) {

    }
    
    func step(accumulatedGradientsData: [MPSGraphTensorData], onlyUseParamsGreaterThan: Int = -1) {
        
    }
}


class AdamOptimizer: Optimizer {
    let ENV: IAEnvironment
    
    let lr: Double
    let beta1: Double
    let beta2: Double
    let epsilon: Double

    // Input placeholders
    var valuesPlaceholders: [MPSGraphTensor] = []
    var gradientPlaceholders: [MPSGraphTensor] = []
    var momentsPlaceholders: [MPSGraphTensor] = []
    var velocitiesPlaceholders: [MPSGraphTensor] = []
    
    // Stored moment + velocity data
    var momentsData: [MPSGraphTensorData] = []
    var velocitiesData: [MPSGraphTensorData] = []

    // Input feeds
    var feeds: [MPSGraphTensor : MPSGraphTensorData] = [:]
    
    // Output targets
    var valuesOutputs: [MPSGraphTensor] = []
    var momentsOutputs: [MPSGraphTensor] = []
    var velocitiesOutputs: [MPSGraphTensor] = []
    var allOutputs: [MPSGraphTensor] = []

    // Adam graph + params
    let graph: MPSGraph;
    let lrTensor: MPSGraphTensor;
    let beta1Tensor: MPSGraphTensor;
    let beta2Tensor: MPSGraphTensor;
    let epsilonTensor: MPSGraphTensor;
    
    init(_ ENV: IAEnvironment, lr: Double = 1e-3, beta1: Double = 0.9, beta2: Double = 0.999, epsilon: Double = 1e-8) {
        self.ENV = ENV
        
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.graph = MPSGraph()
        self.lrTensor = graph.constant(lr, shape: [], dataType: .float32)
        self.beta1Tensor = graph.constant(beta1, shape: [], dataType: .float32)
        self.beta2Tensor = graph.constant(beta2, shape: [], dataType: .float32)
        self.epsilonTensor = graph.constant(epsilon, shape: [], dataType: .float32)
    }
    
    override func setup(param_ids: [Int], NOF_MICROBATCHES: Int) {
        // should only be run when slate has been cleared
        
        // creates zeroed moments + velocities, graph, and input + output placeholders for each param id provided
        self.param_ids.append(contentsOf: param_ids)
        
        var idx = 0
        for param_id in param_ids {
            self.PARAM_ID_TO_IDX[param_id] = idx
            idx += 1

            self.valuesPlaceholders.append(graph.placeholder(shape: ENV.parameterHandler.getParameterShape(sendableID: param_id), dataType: .float32, name: "p_v\(param_id)"))
            self.gradientPlaceholders.append(graph.placeholder(shape: ENV.parameterHandler.getParameterShape(sendableID: param_id), dataType: .float32, name: "p_g\(param_id)"))
            self.momentsPlaceholders.append(graph.placeholder(shape: ENV.parameterHandler.getParameterShape(sendableID: param_id), dataType: .float32, name: "p_m\(param_id)"))
            self.velocitiesPlaceholders.append(graph.placeholder(shape: ENV.parameterHandler.getParameterShape(sendableID: param_id), dataType: .float32, name: "p_vel\(param_id)"))
            self.momentsData.append(MobilePipeUtils.zerosTensorData(ENV, shape: ENV.parameterHandler.getParameterShape(sendableID: param_id)))
            self.velocitiesData.append(MobilePipeUtils.zerosTensorData(ENV, shape: ENV.parameterHandler.getParameterShape(sendableID: param_id)))
            
            let scaleTensor = graph.constant(Double(NOF_MICROBATCHES), shape: [], dataType: .float32)
            let scaledGradientTensor = graph.division(gradientPlaceholders[gradientPlaceholders.count - 1], scaleTensor, name: "scale_\(param_id)")

            let adamOutput = graph.adam(
                currentLearningRate: lrTensor,
                beta1: beta1Tensor,
                beta2: beta2Tensor,
                epsilon: epsilonTensor,
                values: valuesPlaceholders[valuesPlaceholders.count - 1],
                momentum: momentsPlaceholders[momentsPlaceholders.count - 1],
                velocity: velocitiesPlaceholders[velocitiesPlaceholders.count - 1],
                maximumVelocity: nil,
                gradient: scaledGradientTensor,
                name: "adam_\(param_id)"
            )
            
            let (updatedValue, updatedMoment, updatedVelocity) = (adamOutput[0], adamOutput[1], adamOutput[2])
            self.valuesOutputs.append(updatedValue)
            self.momentsOutputs.append(updatedMoment)
            self.velocitiesOutputs.append(updatedVelocity)
            allOutputs.append(updatedValue)
            allOutputs.append(updatedMoment)
            allOutputs.append(updatedVelocity)
        }
    }

    override func step(accumulatedGradientsData: [MPSGraphTensorData], onlyUseParamsGreaterThan: Int = -1) {
        // updates moments & velocities MPSGraphTensorData, updates parameter MPSGraphTensorData in `ENV.parameterHandler`
        var a = 0
        for i in 0..<self.param_ids.count {
            if self.param_ids[i] <= onlyUseParamsGreaterThan  {
                feeds.removeValue(forKey: valuesPlaceholders[i])
                feeds.removeValue(forKey: gradientPlaceholders[i])
                feeds.removeValue(forKey: momentsPlaceholders[i])
                feeds.removeValue(forKey: velocitiesPlaceholders[i])
                continue
            }
            feeds[valuesPlaceholders[i]] = ENV.parameterHandler.getParameterTensorData(sendableID: self.param_ids[i])
            feeds[gradientPlaceholders[i]] = accumulatedGradientsData[a]
            feeds[momentsPlaceholders[i]] = momentsData[i]
            feeds[velocitiesPlaceholders[i]] = velocitiesData[i]
            a += 1
        }
        let results = graph.run(feeds: feeds, targetTensors: Array(allOutputs[((self.param_ids.count - a) * 3)..<allOutputs.count]), targetOperations: nil)
        for i in 0..<self.param_ids.count {
            if self.param_ids[i] <= onlyUseParamsGreaterThan  {
                continue
            }
            ENV.parameterHandler.setParameterTensorData(sendableID: self.param_ids[i], data: results[valuesOutputs[i]]!)
            momentsData[i] = results[momentsOutputs[i]]!
            velocitiesData[i] = results[velocitiesOutputs[i]]!
        }
    }
}


class SGDOptimizer: Optimizer {
    let ENV: IAEnvironment

    var valuesPlaceholders: [MPSGraphTensor] = []
    var gradientPlaceholders: [MPSGraphTensor] = []
    
    var feeds: [MPSGraphTensor : MPSGraphTensorData] = [:]
    
    var valuesOutputs: [MPSGraphTensor] = []

    let graph: MPSGraph;
    let lrTensor: MPSGraphTensor;
    
    init(_ ENV: IAEnvironment, lr: Double = 1e-3) {
        self.ENV = ENV
        self.graph = MPSGraph()
        self.lrTensor = graph.constant(lr, shape: [], dataType: .float32)
    }
    
    override func setup(param_ids: [Int], NOF_MICROBATCHES: Int) {
        
        // creates zeroed moments + velocities, graph, and input + output placeholders for each param id provided
        self.param_ids.append(contentsOf: param_ids)
        
        var idx = 0
        for param_id in param_ids {
            self.PARAM_ID_TO_IDX[param_id] = idx
            idx += 1

            self.valuesPlaceholders.append(graph.placeholder(shape: ENV.parameterHandler.getParameterShape(sendableID: param_id), dataType: .float32, name: "p_v\(param_id)"))
            let valueTensor = self.valuesPlaceholders[self.valuesPlaceholders.count - 1]
            self.gradientPlaceholders.append(graph.placeholder(shape: ENV.parameterHandler.getParameterShape(sendableID: param_id), dataType: .float32, name: "p_g\(param_id)"))
            let gradientTensor = self.gradientPlaceholders[self.gradientPlaceholders.count - 1]
            
            let scaleTensor = graph.constant(Double(NOF_MICROBATCHES), shape: [], dataType: .float32)
            let scaledGradientTensor = graph.division(gradientTensor, scaleTensor, name: "scale_\(param_id)")
            
//            let sgdOutput = graph.stochasticGradientDescent(learningRate: lrTensor,
//                                                             values: valuesPlaceholders[valuesPlaceholders.count - 1],
//                                                             gradient: scaledGradientTensor, name: "SGD")
            let a = graph.multiplication(scaledGradientTensor, lrTensor, name: "SGD_mul")
            let sgdOutput = graph.subtraction(valueTensor, a, name: "SGD_sub")
            
            self.valuesOutputs.append(sgdOutput)
        }
    }

    override func step(accumulatedGradientsData: [MPSGraphTensorData], onlyUseParamsGreaterThan: Int = -1) {
        // updates moments & velocities MPSGraphTensorData, updates parameter MPSGraphTensorData in `ENV.parameterHandler`
        var a = 0
        for i in 0..<self.param_ids.count {
            if self.param_ids[i] <= onlyUseParamsGreaterThan  {
                feeds.removeValue(forKey: valuesPlaceholders[i])
                feeds.removeValue(forKey: gradientPlaceholders[i])
                continue
            }
            feeds[valuesPlaceholders[i]] = ENV.parameterHandler.getParameterTensorData(sendableID: self.param_ids[i])
            feeds[gradientPlaceholders[i]] = accumulatedGradientsData[a]
            a += 1
        }
        let results = graph.run(feeds: feeds, targetTensors: Array(valuesOutputs[((self.param_ids.count - a))..<valuesOutputs.count]), targetOperations: nil)
        for i in 0..<self.param_ids.count {
            if self.param_ids[i] <= onlyUseParamsGreaterThan  {
                continue
            }
            ENV.parameterHandler.setParameterTensorData(sendableID: self.param_ids[i], data: results[valuesOutputs[i]]!)
        }
    }
}


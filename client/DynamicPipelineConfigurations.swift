import MetalPerformanceShadersGraph


class DynamicPipelineConfigurations {
    static func C1(_ ENV: IAEnvironment, _ NOF_DYNAMIC_TEST_BATCHES: Int, _ NOF_MICROBATCHES: Int, _ SCALE_LR: Int) async throws -> Optimizer {
        print("STARTING C1")

        let graph_PreL2 = MPSGraph()
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        var paramPlaceholders: [MPSGraphTensor] = []
        var paramsToTrain: [MPSGraphTensor] = []
        var trainedParamIds: [Int] = []

        for id in 1..<MobilePipeUtils.RESNET34_S2_NOF_PARAMS + 1 {
            let paramPlaceholder = MobilePipeUtils.placeHolderFromId(ENV, graph_PreL2, id)
            paramPlaceholders.append(paramPlaceholder)
            feeds[paramPlaceholders[id - 1]] = ENV.parameterHandler.getParameterTensorData(sendableID: id)

            if MobilePipeUtils.paramIdIsTrainable(id) {
                trainedParamIds.append(id)
                paramsToTrain.append(paramPlaceholder)
            }
        }
        
        var graphBuilt = false
        
        var inputPlaceholder: MPSGraphTensor?, targetPlaceholder: MPSGraphTensor?
        var forwardOutputPlaceholder: MPSGraphTensor?, lossPlaceholder: MPSGraphTensor?
        var gradientsCalculationOutput: [MPSGraphTensor: MPSGraphTensor]? = nil
        var currentAccumulatedGradientsPlaceholders: [MPSGraphTensor] = []
        var accumulatedGradientData: [MPSGraphTensorData] = []
        var newAccumulatedGradientsPlaceholders: [MPSGraphTensor] = []
        var targetTensors: [MPSGraphTensor] = []
        
//        let optimizer = AdamOptimizer(ENV)
        let optimizer: SGDOptimizer = SGDOptimizer(ENV, lr: 1e-3 * Double(SCALE_LR))
        optimizer.setup(param_ids: trainedParamIds, NOF_MICROBATCHES: NOF_MICROBATCHES)
        
        ENV.TIMEK.clear()
        
        for _ in 0..<NOF_DYNAMIC_TEST_BATCHES {
            let NOF_MICROBATCHES = try await ENV.commHandler.receiveUInt8()  // nof microbatches per batch
            
            Task.detached {
                // parallel comms
                for i in 0..<NOF_MICROBATCHES {  // input, target
//                    ENV.log("Awaiting:" + String(i))
                    let input = try await ENV.commHandler.receiveSendableTensor()
                    let target = try await ENV.commHandler.receiveSendableTensor()
                    await MobilePipeUtils.GlobalTensorQueue.push(input)
                    await MobilePipeUtils.GlobalTensorQueue.push(target)
                }
            }
            
            for i in 0..<NOF_MICROBATCHES {
                ENV.TIMEK.start(cat: TimeKeeper.CATS.WAITING_FOR_INPUT)
                let input = await MobilePipeUtils.GlobalTensorQueue.pop()
                let target = await MobilePipeUtils.GlobalTensorQueue.pop()
                ENV.TIMEK.end(cat: TimeKeeper.CATS.WAITING_FOR_INPUT)

                ENV.TIMEK.start(cat: TimeKeeper.CATS.S2_COMBINED_MICROBATCH)
                if !graphBuilt {
                    inputPlaceholder = ENV.graphHandler.makePlaceholderFloat32Tensor(graph: graph_PreL2, shape: input.dims, name: "input")
                    targetPlaceholder = ENV.graphHandler.makePlaceholderFloat32Tensor(graph: graph_PreL2, shape: target.dims, name: "target")
                    
                    forwardOutputPlaceholder = ENV.graphHandler.ResNet_S2_PreLayer2(graph: graph_PreL2, input: inputPlaceholder!, params: paramPlaceholders, frozenRunningParams: true)
                    
                    lossPlaceholder = ENV.graphHandler.CrossEntropy(graph: graph_PreL2,
                                                                    input: forwardOutputPlaceholder!,
                                                                    target: targetPlaceholder!,
                                                                    name: "loss")

                    var tensorsRequiringGrads = paramsToTrain
                    tensorsRequiringGrads.append(inputPlaceholder!)
                    gradientsCalculationOutput = graph_PreL2.gradients(of: lossPlaceholder!, with: tensorsRequiringGrads, name: "grads")
                    gradientsCalculationOutput!.forEach { _, gradient in
                        targetTensors.append(gradient)
                    }
                    
                    for i in 0..<trainedParamIds.count {
                        // initialize accumulatedGradients to zero
                        let paramShape = ENV.parameterHandler.getParameterShape(sendableID: trainedParamIds[i])
                        accumulatedGradientData.append(MobilePipeUtils.zerosTensorData(ENV, shape: paramShape))
                        currentAccumulatedGradientsPlaceholders.append(ENV.graphHandler.makePlaceholderFloat32Tensor(graph: graph_PreL2,
                                                                                                                     shape: paramShape,
                                                                                                                     name: "currentAccumulatedGradient:\(i)"))
                        newAccumulatedGradientsPlaceholders.append(graph_PreL2.addition(currentAccumulatedGradientsPlaceholders[i],
                                                                                        gradientsCalculationOutput![paramsToTrain[i]]!,
                                                                                        name: "newAccumlatedGradient:\(i)"))
                    }
                    targetTensors.append(contentsOf: newAccumulatedGradientsPlaceholders)
                    targetTensors.append(lossPlaceholder!)
                    graphBuilt = true
                }
                
                // update feeds
                feeds[inputPlaceholder!] = ENV.graphHandler.makeTensorData(input)
                feeds[targetPlaceholder!] = ENV.graphHandler.makeTensorData(target)
                for i in 0..<trainedParamIds.count {
                    feeds[currentAccumulatedGradientsPlaceholders[i]] = accumulatedGradientData[i]
                }

                // run
                let results = graph_PreL2.run(feeds: feeds, targetTensors: targetTensors, targetOperations: nil)

                // send gradient & loss to s1_backward
                let r = results[gradientsCalculationOutput![inputPlaceholder!]!]!
                ENV.commHandler.sendSendableTensor(tensor: DataHandler.toSendableFloat32Tensor(r))
                ENV.commHandler.sendSendableTensor(tensor: DataHandler.toSendableFloat32Tensor(results[lossPlaceholder!]!))
//                ENV.log("Sent:" + String(i))
                
                // update accumulated gradients
                for i in 0..<accumulatedGradientData.count {
                    accumulatedGradientData[i] = results[newAccumulatedGradientsPlaceholders[i]]!
                }
                ENV.TIMEK.end(cat: TimeKeeper.CATS.S2_COMBINED_MICROBATCH)
            }
                
            ENV.TIMEK.start(cat: TimeKeeper.CATS.OPTIMIZATION_STEP)
//            optimizer.step(accumulatedGradientsData: accumulatedGradientData)
            // zero grad
            for i in 0..<accumulatedGradientData.count {
                accumulatedGradientData[i] = MobilePipeUtils.zerosTensorData(ENV, shape: accumulatedGradientData[i].shape)
            }
            
            // update G1_feeds[all G1_placeholders] = all Adam-ized weights
            for id in trainedParamIds {
                feeds[paramPlaceholders[id - 1]] = ENV.parameterHandler.getParameterTensorData(sendableID: id)
            }
            ENV.TIMEK.end(cat: TimeKeeper.CATS.OPTIMIZATION_STEP)
        }
        
//         SYNC
//        try await DynamicStageSync.SYNC_POST_C1(ENV, optimizer: optimizer)
        
        // SEND IOS TIMES
        ENV.TIMEK.send_ios_times(ENV.commHandler)
        ENV.TIMEK.clear()

        return optimizer
    }
    
    
    static func C2(_ ENV: IAEnvironment, _ NOF_DYNAMIC_TEST_BATCHES: Int, _ optimizer: Optimizer) async throws -> Optimizer {
        print("STARTING C2")
        
        let c = ENV.commHandler, g = ENV.graphHandler, p = ENV.parameterHandler
        
        let NOF_PARAMS = MobilePipeUtils.RESNET34_S2_NOF_PARAMS  // all params for: layer 2 + layer3 + layer4 + end
        let NOF_L2_PARAMS = MobilePipeUtils.RESNET34_L2_PARAMS

        let G2_PreL3_GRAPH = MPSGraph()
        var G2_placeholders: [MPSGraphTensor] = []
        var G2_targetGrads: [MPSGraphTensor] = []
        var G2_feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]  //      L3 + L4

        var TRAINABLE_PARAM_IDS: [Int] = []  // `.count` should be 31 from model, then we add `input` for 32
        
        for id in 1..<NOF_PARAMS + 1 {
            let p2 = MobilePipeUtils.placeHolderFromId(ENV, G2_PreL3_GRAPH, id)
            
            if MobilePipeUtils.paramIdIsTrainable(id) {  // training only convs weight + fc weight & bias
                if id > NOF_L2_PARAMS {  // in L3 + L4
                    TRAINABLE_PARAM_IDS.append(id)
                    G2_targetGrads.append(p2)
                }
            }
            
            if id > NOF_L2_PARAMS {  // if in L3 + L4
                G2_placeholders.append(p2)
                G2_feeds[G2_placeholders[id - NOF_L2_PARAMS - 1]] = p.getParameterTensorData(sendableID: id)
            }
        }
        
        // handling efficient placeholder and graph creation
        var placeholdersBuilt = false
        
        var inputTensor: MPSGraphTensor? = nil
        var targetTensor: MPSGraphTensor? = nil
        var resultTensor: MPSGraphTensor? = nil
        var lossTensor: MPSGraphTensor? = nil
        var gradientsOutput: [MPSGraphTensor: MPSGraphTensor]? = nil
        var prevAccumulatedGradients: [MPSGraphTensor] = []
        var accumulatedGradientData: [MPSGraphTensorData] = []
        var accumulatedGradientOutputs: [MPSGraphTensor] = []
        var targetTensors: [MPSGraphTensor] = []
        
        let tensorQueue = MobilePipeUtils.GlobalTensorQueue
        
        // === G2 TEST ===
        for _ in 0..<NOF_DYNAMIC_TEST_BATCHES {
            let NOF_MICROBATCHES = try await c.receiveUInt8()  // nof microbatches per batch
            Task.detached {
                for _ in 0..<NOF_MICROBATCHES {  // input, target
                    let input = try await ENV.commHandler.receiveSendableTensor()
                    let target = try await ENV.commHandler.receiveSendableTensor()
                    await tensorQueue.push(input)
                    await tensorQueue.push(target)
                }
            }
            for _ in 0..<NOF_MICROBATCHES {
                ENV.TIMEK.start(cat: TimeKeeper.CATS.WAITING_FOR_INPUT)
                let input = await tensorQueue.pop()
                let target = await tensorQueue.pop()
                ENV.TIMEK.end(cat: TimeKeeper.CATS.WAITING_FOR_INPUT)

                ENV.TIMEK.start(cat: TimeKeeper.CATS.S2_COMBINED_MICROBATCH)
                // build placeholders and graph if needed
                if !placeholdersBuilt {
                    inputTensor = ENV.graphHandler.makePlaceholderFloat32Tensor(graph: G2_PreL3_GRAPH, shape: input.dims, name: "Input")
                    targetTensor = ENV.graphHandler.makePlaceholderFloat32Tensor(graph: G2_PreL3_GRAPH, shape: target.dims, name: "Target")

                    resultTensor = g.ResNet_S2_PreLayer3(graph: G2_PreL3_GRAPH, input: inputTensor!, params: G2_placeholders, frozenRunningParams: true)

                    lossTensor = g.CrossEntropy(graph: G2_PreL3_GRAPH, input: resultTensor!, target: targetTensor!, name: "Loss")

                    targetTensors = []

                    G2_targetGrads.append(inputTensor!)
                    gradientsOutput = G2_PreL3_GRAPH.gradients(of: lossTensor!, with: G2_targetGrads, name: "Gradients")
                    for i in 0..<G2_targetGrads.count {
                        // add all target grads to graph result
                        targetTensors.append(gradientsOutput![G2_targetGrads[i]]!)
                    }

                    var i = 0
                    for a in 0..<TRAINABLE_PARAM_IDS.count {
                        if TRAINABLE_PARAM_IDS[a] <= NOF_L2_PARAMS {
                            continue
                        }
                        // initialize accumulatedGradients to zero
                        let shape = ENV.parameterHandler.getParameterShape(sendableID: TRAINABLE_PARAM_IDS[a])
                        accumulatedGradientData.append(MobilePipeUtils.zerosTensorData(ENV, shape: shape))
                        prevAccumulatedGradients.append(ENV.graphHandler.makePlaceholderFloat32Tensor(graph: G2_PreL3_GRAPH, shape: shape, name: "prevAcumulatedGrad:\(i)"))
                        accumulatedGradientOutputs.append(G2_PreL3_GRAPH.addition(prevAccumulatedGradients[i],
                                                                                  gradientsOutput![G2_targetGrads[i]]!,
                                                                                  name: "newAccumlatedGrad:\(i)"))  // curSumGrad + newGrad
                        i += 1
                    }
                    targetTensors.append(contentsOf: accumulatedGradientOutputs)

                    placeholdersBuilt = true
                }
                targetTensors.append(lossTensor!)

                // update feeds
                G2_feeds[inputTensor!] = ENV.graphHandler.makeTensorData(input)
                G2_feeds[targetTensor!] = ENV.graphHandler.makeTensorData(target)
                for i in 0..<prevAccumulatedGradients.count {
                    G2_feeds[prevAccumulatedGradients[i]] = accumulatedGradientData[i]
                }

                // run
                let results = G2_PreL3_GRAPH.run(feeds: G2_feeds, targetTensors: targetTensors, targetOperations: nil)

                // send gradient & loss to s1_backward
                let r = results[gradientsOutput![inputTensor!]!]!
                let d = DataHandler.toSendableFloat32Tensor(r)
                c.sendSendableTensor(tensor: d)
                c.sendSendableTensor(tensor: DataHandler.toSendableFloat32Tensor(results[lossTensor!]!))

                // update accumulated gradients
                for i in 0..<accumulatedGradientData.count {
                    accumulatedGradientData[i] = results[accumulatedGradientOutputs[i]]!
                }
                ENV.TIMEK.end(cat: TimeKeeper.CATS.S2_COMBINED_MICROBATCH)
            }

            ENV.TIMEK.start(cat: TimeKeeper.CATS.OPTIMIZATION_STEP)
//            optimizer.step(accumulatedGradientsData: accumulatedGradientData, onlyUseParamsGreaterThan: NOF_L2_PARAMS)
            // zero grad
            for i in 0..<accumulatedGradientData.count {
                accumulatedGradientData[i] = MobilePipeUtils.zerosTensorData(ENV, shape: accumulatedGradientData[i].shape)
            }

            // update G2_feeds[all G2_placeholders] = all Adam-ized weights
            for i in 0..<TRAINABLE_PARAM_IDS.count {
                if TRAINABLE_PARAM_IDS[i] <= NOF_L2_PARAMS {
                    continue
                }
                G2_feeds[G2_placeholders[TRAINABLE_PARAM_IDS[i] - 1 - NOF_L2_PARAMS]] = p.getParameterTensorData(sendableID: TRAINABLE_PARAM_IDS[i])
            }
            ENV.TIMEK.end(cat: TimeKeeper.CATS.OPTIMIZATION_STEP)
        }

//         SYNC
//        try await DynamicStageSync.SYNC_POST_C2(ENV, optimizer: optimizer)

        // SEND IOS TIMES
        ENV.TIMEK.send_ios_times(ENV.commHandler)
        ENV.TIMEK.clear()

        return optimizer
    }
    
    
    static func C3(_ ENV: IAEnvironment, _ NOF_DYNAMIC_TEST_BATCHES: Int, _ optimizer: Optimizer) async throws -> Optimizer {
        print("STARTING C3")
        
        let c = ENV.commHandler, g = ENV.graphHandler, p = ENV.parameterHandler
        
        let NOF_PARAMS = MobilePipeUtils.RESNET34_S2_NOF_PARAMS  // all params for: layer 2 + layer3 + layer4 + end
        let NOF_L2_PARAMS = MobilePipeUtils.RESNET34_L2_PARAMS
        let NOF_L3_PARAMS = MobilePipeUtils.RESNET34_L3_PARAMS

        let C3_PreL4_GRAPH = MPSGraph()
        var C3_placeholders: [MPSGraphTensor] = []
        var C3_targetGrads: [MPSGraphTensor] = []
        var C3_feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]  //           L4

        var TRAINABLE_PARAM_IDS: [Int] = []  // `.count` should be 31 from model, then we add `input` for 32
        for i in 1..<NOF_PARAMS + 1 {
            if (i - 1) % 5 == 0 || i == NOF_PARAMS - 1 || i == NOF_PARAMS {
                TRAINABLE_PARAM_IDS.append(i)
            }
        }
        
        // adding placeholders to graphs, adding target grads
        for id in 1..<NOF_PARAMS + 1 {
            let p3 = MobilePipeUtils.placeHolderFromId(ENV, C3_PreL4_GRAPH, id)
            if id > NOF_L2_PARAMS + NOF_L3_PARAMS {
                C3_placeholders.append(p3)
            }
            
            if (id - 1) % 5 == 0 || id == NOF_PARAMS - 1 || id == NOF_PARAMS {  // training only convs weight + fc weight & bias
                if id > NOF_L2_PARAMS + NOF_L3_PARAMS {  // in L4
                    C3_targetGrads.append(p3)
                }
            }
        }
        
        // creating feeds for graphs
        for id in 1..<NOF_PARAMS + 1 {
            if id > NOF_L2_PARAMS + NOF_L3_PARAMS {  // if in L4
                C3_feeds[C3_placeholders[id - NOF_L3_PARAMS - NOF_L2_PARAMS - 1]] = p.getParameterTensorData(sendableID: id)
            }
        }
        
        // handling efficient placeholder and graph creation
        var placeholdersBuilt = false
        
        var inputTensor: MPSGraphTensor? = nil
        var targetTensor: MPSGraphTensor? = nil
        var resultTensor: MPSGraphTensor? = nil
        var lossTensor: MPSGraphTensor? = nil
        var gradientsOutput: [MPSGraphTensor: MPSGraphTensor]? = nil
        var prevAccumulatedGradients: [MPSGraphTensor] = []
        var accumulatedGradientData: [MPSGraphTensorData] = []
        var accumulatedGradientOutputs: [MPSGraphTensor] = []
        var targetTensors: [MPSGraphTensor] = []
        
        let tensorQueue = MobilePipeUtils.GlobalTensorQueue
        
        // === G3 TEST ===
        for _ in 0..<NOF_DYNAMIC_TEST_BATCHES {
            let NOF_MICROBATCHES = try await c.receiveUInt8()  // nof microbatches per batch
            Task.detached {
                for _ in 0..<NOF_MICROBATCHES {  // input, target
                    let input = try await ENV.commHandler.receiveSendableTensor()
                    let target = try await ENV.commHandler.receiveSendableTensor()
                    await tensorQueue.push(input)
                    await tensorQueue.push(target)
                }
            }
            for _ in 0..<NOF_MICROBATCHES {
                ENV.TIMEK.start(cat: TimeKeeper.CATS.WAITING_FOR_INPUT)
                let input = await tensorQueue.pop()
                let target = await tensorQueue.pop()
                ENV.TIMEK.end(cat: TimeKeeper.CATS.WAITING_FOR_INPUT)

                ENV.TIMEK.start(cat: TimeKeeper.CATS.S2_COMBINED_MICROBATCH)
                // build placeholders and graph if needed
                if !placeholdersBuilt {
                    inputTensor = ENV.graphHandler.makePlaceholderFloat32Tensor(graph: C3_PreL4_GRAPH, shape: input.dims, name: "Input")
                    targetTensor = ENV.graphHandler.makePlaceholderFloat32Tensor(graph: C3_PreL4_GRAPH, shape: target.dims, name: "Target")

                    resultTensor = g.ResNet_S2_PreLayer4(graph: C3_PreL4_GRAPH, input: inputTensor!, params: C3_placeholders, frozenRunningParams: true)

                    lossTensor = g.CrossEntropy(graph: C3_PreL4_GRAPH, input: resultTensor!, target: targetTensor!, name: "Loss")
                    C3_targetGrads.append(inputTensor!)
                    gradientsOutput = C3_PreL4_GRAPH.gradients(of: lossTensor!, with: C3_targetGrads, name: "Gradients")
                    for i in 0..<C3_targetGrads.count {
                        // add all target grads to graph result
                        targetTensors.append(gradientsOutput![C3_targetGrads[i]]!)
                    }

                    var i = 0
                    for a in 0..<TRAINABLE_PARAM_IDS.count {
                        if TRAINABLE_PARAM_IDS[a] <= NOF_L2_PARAMS + NOF_L3_PARAMS {
                            continue
                        }
                        // initialize accumulatedGradients to zero
                        let shape = ENV.parameterHandler.getParameterShape(sendableID: TRAINABLE_PARAM_IDS[a])
                        accumulatedGradientData.append(MobilePipeUtils.zerosTensorData(ENV, shape: shape))
                        prevAccumulatedGradients.append(ENV.graphHandler.makePlaceholderFloat32Tensor(graph: C3_PreL4_GRAPH, shape: shape, name: "prevAcumulatedGrad:\(i)"))
                        accumulatedGradientOutputs.append(C3_PreL4_GRAPH.addition(prevAccumulatedGradients[i],
                                                                                  gradientsOutput![C3_targetGrads[i]]!,
                                                                                  name: "newAccumlatedGrad:\(i)"))  // curSumGrad + newGrad
                        i += 1
                    }
                    targetTensors.append(contentsOf: accumulatedGradientOutputs)

                    placeholdersBuilt = true
                }
                targetTensors.append(lossTensor!)

                // update feeds
                C3_feeds[inputTensor!] = ENV.graphHandler.makeTensorData(input)
                C3_feeds[targetTensor!] = ENV.graphHandler.makeTensorData(target)
                for i in 0..<prevAccumulatedGradients.count {
                    C3_feeds[prevAccumulatedGradients[i]] = accumulatedGradientData[i]
                }

                // run
                let results = C3_PreL4_GRAPH.run(feeds: C3_feeds, targetTensors: targetTensors, targetOperations: nil)

                // send gradient & loss to s1_backward
                c.sendSendableTensor(tensor: DataHandler.toSendableFloat32Tensor(results[gradientsOutput![inputTensor!]!]!))
                c.sendSendableTensor(tensor: DataHandler.toSendableFloat32Tensor(results[lossTensor!]!))

                // update accumulated gradients
                for i in 0..<accumulatedGradientData.count {
                    accumulatedGradientData[i] = results[accumulatedGradientOutputs[i]]!
                }
                ENV.TIMEK.end(cat: TimeKeeper.CATS.S2_COMBINED_MICROBATCH)
            }

            ENV.TIMEK.start(cat: TimeKeeper.CATS.OPTIMIZATION_STEP)
//            optimizer.step(accumulatedGradientsData: accumulatedGradientData, onlyUseParamsGreaterThan: NOF_L2_PARAMS + NOF_L3_PARAMS)
            // zero grad
            for i in 0..<accumulatedGradientData.count {
                accumulatedGradientData[i] = MobilePipeUtils.zerosTensorData(ENV, shape: accumulatedGradientData[i].shape)
            }

            // update G3_feeds[all G3_placeholders] = all Adam-ized weights
            for i in 0..<TRAINABLE_PARAM_IDS.count {
                if TRAINABLE_PARAM_IDS[i] <= NOF_L2_PARAMS + NOF_L3_PARAMS {
                    continue
                }
                C3_feeds[C3_placeholders[TRAINABLE_PARAM_IDS[i] - 1 - NOF_L2_PARAMS - NOF_L3_PARAMS]] = p.getParameterTensorData(sendableID: TRAINABLE_PARAM_IDS[i])
            }
            ENV.TIMEK.end(cat: TimeKeeper.CATS.OPTIMIZATION_STEP)
        }

        // SEND IOS TIMES
        ENV.TIMEK.send_ios_times(ENV.commHandler)
        ENV.TIMEK.clear()

        return optimizer
    }
}


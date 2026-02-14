import MetalPerformanceShadersGraph


class ContinuedTraining {
    static func C1(_ ENV: IAEnvironment, _ optimizer: Optimizer, adam: Bool = false) async throws {
        // load need weights and adam states if any needed
        var opt_idx = 0  // start from L2 trainables
        while true {
            let id = try await ENV.commHandler.receiveUInt8()
            if id == 0 {
                break
            }
            let param = try await ENV.commHandler.receiveSendableTensor()
            let moment = try await ENV.commHandler.receiveSendableTensor()
            let velocity = try await ENV.commHandler.receiveSendableTensor()

            // store param
            ENV.parameterHandler.storeParameter(sendableID: id, tensor: param)
            
            if adam {
                // store moment, velocity in optimizer
                let o = optimizer as! AdamOptimizer
                o.momentsData[opt_idx] = ENV.graphHandler.makeTensorData(moment)
                o.velocitiesData[opt_idx] = ENV.graphHandler.makeTensorData(velocity)
            }
            
            // will only be receiving trainables for L2 + L3 (conv layers); opt's arrs contain only trainables starting at L2
            opt_idx += 1
        }
        
        // start training
        let c = ENV.commHandler, g = ENV.graphHandler, p = ENV.parameterHandler
        
        let NOF_PARAMS = 147  // all params for: layer 2 + layer3 + layer4 + end

        let C1_PreL2_GRAPH = MPSGraph()
        var C1_feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]  // L2 + L3 + L4
        var C1_placeholders: [MPSGraphTensor] = []
        var C1_targetGrads: [MPSGraphTensor] = []

        // assign trainable params
        var TRAINABLE_PARAM_IDS: [Int] = []  // `.count` should be 31 from model, then we add `input` for 32
        for i in 1..<NOF_PARAMS + 1 {
            if (i - 1) % 5 == 0 || i == NOF_PARAMS - 1 || i == NOF_PARAMS {
                TRAINABLE_PARAM_IDS.append(i)
            }
        }
        
        // adding placeholders to graphs, adding target grads
        for id in 1..<NOF_PARAMS + 1 {
            let p1 = MobilePipeUtils.placeHolderFromId(ENV, C1_PreL2_GRAPH, id)
            C1_placeholders.append(p1)
            if (((id - 1) % 5) == 0 || id == NOF_PARAMS - 1 || id == NOF_PARAMS) {  // training only convs weight + fc weight & bias
                C1_targetGrads.append(p1)
            }
        }
        
        // creating feeds for graphs
        for id in 1..<NOF_PARAMS + 1 {
            C1_feeds[C1_placeholders[id - 1]] = p.getParameterTensorData(sendableID: id)
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

        while true {
            let NOF_MICROBATCHES = try await c.receiveUInt8()  // nof microbatches per batch
            if NOF_MICROBATCHES == 0 {
                break
            }
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
                    inputTensor = ENV.graphHandler.makePlaceholderFloat32Tensor(graph: C1_PreL2_GRAPH, shape: input.dims, name: "Input")
                    targetTensor = ENV.graphHandler.makePlaceholderFloat32Tensor(graph: C1_PreL2_GRAPH, shape: target.dims, name: "Target")
                    
                    resultTensor = g.ResNet_S2_PreLayer2(graph: C1_PreL2_GRAPH, input: inputTensor!, params: C1_placeholders, frozenRunningParams: true)
                    
                    lossTensor = g.CrossEntropy(graph: C1_PreL2_GRAPH, input: resultTensor!, target: targetTensor!, name: "Loss")
                    lossTensor = C1_PreL2_GRAPH.division(lossTensor!,
                                                         C1_PreL2_GRAPH.constant(Double(NOF_MICROBATCHES), dataType: .float32), name: "scaleLoss")  // scale

                    C1_targetGrads.append(inputTensor!)
                    gradientsOutput = C1_PreL2_GRAPH.gradients(of: lossTensor!, with: C1_targetGrads, name: "Gradients")
                    for i in 0..<C1_targetGrads.count {
                        // add all target grads to graph result
                        targetTensors.append(gradientsOutput![C1_targetGrads[i]]!)
                    }
                    
                    for i in 0..<TRAINABLE_PARAM_IDS.count {
                        // initialize accumulatedGradients to zero
                        let shape = ENV.parameterHandler.getParameterShape(sendableID: TRAINABLE_PARAM_IDS[i])
                        accumulatedGradientData.append(MobilePipeUtils.zerosTensorData(ENV, shape: shape))
                        prevAccumulatedGradients.append(ENV.graphHandler.makePlaceholderFloat32Tensor(graph: C1_PreL2_GRAPH, shape: shape, name: "prevAcumulatedGrad:\(i)"))
                        accumulatedGradientOutputs.append(C1_PreL2_GRAPH.addition(prevAccumulatedGradients[i],
                                                                                  gradientsOutput![C1_targetGrads[i]]!,
                                                                                  name: "newAccumlatedGrad:\(i)"))  // curSumGrad + newGrad
                    }
                    targetTensors.append(contentsOf: accumulatedGradientOutputs)
                    targetTensors.append(lossTensor!)
                    placeholdersBuilt = true
                }
                
                // update feeds
                C1_feeds[inputTensor!] = ENV.graphHandler.makeTensorData(input)
                C1_feeds[targetTensor!] = ENV.graphHandler.makeTensorData(target)
                for i in 0..<prevAccumulatedGradients.count {
                    C1_feeds[prevAccumulatedGradients[i]] = accumulatedGradientData[i]
                }

                // run
                let results = C1_PreL2_GRAPH.run(feeds: C1_feeds, targetTensors: targetTensors, targetOperations: nil)
                ENV.TIMEK.end(cat: TimeKeeper.CATS.S2_COMBINED_MICROBATCH)

                // send gradient & loss to s1_backward
                let r = results[gradientsOutput![inputTensor!]!]!
                let d = DataHandler.toSendableFloat32Tensor(r)
                c.sendSendableTensor(tensor: d)
                c.sendSendableTensor(tensor: DataHandler.toSendableFloat32Tensor(results[lossTensor!]!))
                
                // update accumulated gradients
                for i in 0..<accumulatedGradientData.count {
                    accumulatedGradientData[i] = results[accumulatedGradientOutputs[i]]!
                }
            }
                
            ENV.TIMEK.start(cat: TimeKeeper.CATS.OPTIMIZATION_STEP)
            optimizer.step(accumulatedGradientsData: accumulatedGradientData) 
            // zero grad
            for i in 0..<accumulatedGradientData.count {
                accumulatedGradientData[i] = MobilePipeUtils.zerosTensorData(ENV, shape: accumulatedGradientData[i].shape)
            }
            
            // update G1_feeds[all G1_placeholders] = all Adam-ized weights
            for id in TRAINABLE_PARAM_IDS {
                C1_feeds[C1_placeholders[id - 1]] = p.getParameterTensorData(sendableID: id)
            }
            ENV.TIMEK.end(cat: TimeKeeper.CATS.OPTIMIZATION_STEP)
        }
        
        // TODO: sync final
    }
    
    static func C2(_ ENV: IAEnvironment, _ optimizer: Optimizer, adam: Bool = false) async throws {
        print("Starting Continued C2")
        // load need weights and adam states if any needed
        var opt_idx = 0  // start from L3 trainables
        for i in 0..<optimizer.param_ids.count {
            if optimizer.param_ids[i] <= MobilePipeUtils.RESNET34_L2_PARAMS {
                opt_idx += 1
            } else {
                break
            }
        }
        while true {
            let id = try await ENV.commHandler.receiveUInt8()
            if id == 0 {
                break
            }
            let param = try await ENV.commHandler.receiveSendableTensor()

            // store param
            ENV.parameterHandler.storeParameter(sendableID: id, tensor: param)
            
            if adam {
                let moment = try await ENV.commHandler.receiveSendableTensor()
                let velocity = try await ENV.commHandler.receiveSendableTensor()
                let o = optimizer as! AdamOptimizer
                // store moment, velocity in optimizer
                o.momentsData[opt_idx] = ENV.graphHandler.makeTensorData(moment)
                o.velocitiesData[opt_idx] = ENV.graphHandler.makeTensorData(velocity)
            }
            
            // will only be receiving trainables for L3 (conv layers); opt's arrs contain only trainables starting at L2
            opt_idx += 1
        }
        
        // start training
        let c = ENV.commHandler, g = ENV.graphHandler, p = ENV.parameterHandler
        
        let NOF_PARAMS = MobilePipeUtils.RESNET34_S2_NOF_PARAMS  // all params for: layer 2 + layer3 + layer4 + end
        let NOF_L2_PARAMS = MobilePipeUtils.RESNET34_L2_PARAMS

        let G2_PreL3_GRAPH = MPSGraph()
        var G2_placeholders: [MPSGraphTensor] = []
        var G2_paramsTargetedForGrads: [MPSGraphTensor] = []
        var G2_feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]  //      L3 + L4

        var TRAINABLE_PARAM_IDS: [Int] = []  // `.count` should be 31 from model, then we add `input` for 32
        
        for id in 1..<NOF_PARAMS + 1 {
            let p2 = MobilePipeUtils.placeHolderFromId(ENV, G2_PreL3_GRAPH, id)
            if id > NOF_L2_PARAMS {  // in L3 + L4
                G2_placeholders.append(p2)
                if MobilePipeUtils.paramIdIsTrainable(id) {  // training only convs weight + fc weight & bias
                    TRAINABLE_PARAM_IDS.append(id)
                    G2_paramsTargetedForGrads.append(p2)
                }
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
        
        while true {
            let NOF_MICROBATCHES = try await c.receiveUInt8()  // nof microbatches per batch
            if NOF_MICROBATCHES == 0 {
                break
            }
            Task.detached {
                for _ in 0..<NOF_MICROBATCHES {  // input, target
                    let input = try await c.receiveSendableTensor()
                    let target = try await c.receiveSendableTensor()
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

                    targetTensors.removeAll()

                    G2_paramsTargetedForGrads.append(inputTensor!)
                    gradientsOutput = G2_PreL3_GRAPH.gradients(of: lossTensor!, with: G2_paramsTargetedForGrads, name: "Gradients")
                    for i in 0..<G2_paramsTargetedForGrads.count {
                        // add all target grads to graph result
                        targetTensors.append(gradientsOutput![G2_paramsTargetedForGrads[i]]!)
                    }

                    var i = 0
                    for a in 0..<TRAINABLE_PARAM_IDS.count {
                        if TRAINABLE_PARAM_IDS[a] <= NOF_L2_PARAMS {
                            continue
                        }
                        let shape = ENV.parameterHandler.getParameterShape(sendableID: TRAINABLE_PARAM_IDS[a])
                        accumulatedGradientData.append(MobilePipeUtils.zerosTensorData(ENV, shape: shape))
                        prevAccumulatedGradients.append(ENV.graphHandler.makePlaceholderFloat32Tensor(graph: G2_PreL3_GRAPH,
                                                                                                      shape: shape,
                                                                                                      name: "prevAcumulatedGrad:\(i)"))
                        accumulatedGradientOutputs.append(G2_PreL3_GRAPH.addition(prevAccumulatedGradients[i],
                                                                                  gradientsOutput![G2_paramsTargetedForGrads[i]]!,
                                                                                  name: "newAccumlatedGradient:\(i)"))
                        i += 1
                    }
                    targetTensors.append(contentsOf: accumulatedGradientOutputs)
                    targetTensors.append(lossTensor!)

                    placeholdersBuilt = true
                }

                // update feeds
                G2_feeds[inputTensor!] = ENV.graphHandler.makeTensorData(input)
                G2_feeds[targetTensor!] = ENV.graphHandler.makeTensorData(target)
                for i in 0..<prevAccumulatedGradients.count {
                    G2_feeds[prevAccumulatedGradients[i]] = accumulatedGradientData[i]
                }

                // run
                let results = G2_PreL3_GRAPH.run(feeds: G2_feeds, targetTensors: targetTensors, targetOperations: nil)

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
            optimizer.step(accumulatedGradientsData: accumulatedGradientData, onlyUseParamsGreaterThan: NOF_L2_PARAMS)
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
        
        // sync final
        for i in 0..<TRAINABLE_PARAM_IDS.count {
            if TRAINABLE_PARAM_IDS[i] <= NOF_L2_PARAMS {
                continue
            }
            c.sendDouble(Double(TRAINABLE_PARAM_IDS[i]))
            c.sendSendableTensor(tensor: DataHandler.toSendableFloat32Tensor(p.getParameterTensorData(sendableID: TRAINABLE_PARAM_IDS[i])))
        }
        c.sendDouble(0)
    }
    
    static func C3(_ ENV: IAEnvironment, _ optimizer: Optimizer) async throws {
        let _ = try await ENV.commHandler.receiveUInt8()  // nothing to sync
        
        let c = ENV.commHandler, g = ENV.graphHandler, p = ENV.parameterHandler
        
        let NOF_PARAMS = MobilePipeUtils.RESNET34_S2_NOF_PARAMS  // all params for: layer 2 + layer3 + layer4 + end
        let NOF_L2_PARAMS = MobilePipeUtils.RESNET34_L2_PARAMS
        let NOF_L3_PARAMS = MobilePipeUtils.RESNET34_L3_PARAMS

        let C3_PreL4_GRAPH = MPSGraph()
        var C3_paramPlaceholders: [MPSGraphTensor] = []
        var C3_placeholdersToTargetForGradients: [MPSGraphTensor] = []
        var C3_feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]  //           L4

        var TRAINABLE_PARAM_IDS: [Int] = []  // `.count` should be 31 from model, then we add `input` for 32
        
        for id in 1..<NOF_PARAMS + 1 {
            let p3 = MobilePipeUtils.placeHolderFromId(ENV, C3_PreL4_GRAPH, id)
            if id > NOF_L2_PARAMS + NOF_L3_PARAMS {
                C3_paramPlaceholders.append(p3)
                if MobilePipeUtils.paramIdIsTrainable(id) {
                    TRAINABLE_PARAM_IDS.append(id)
                    C3_placeholdersToTargetForGradients.append(p3)
                }
                C3_feeds[C3_paramPlaceholders[id - NOF_L3_PARAMS - NOF_L2_PARAMS - 1]] = p.getParameterTensorData(sendableID: id)
            }
        }
        
        // handling efficient placeholder and graph creation
        var placeholdersBuilt = false
        
        var inputTensor: MPSGraphTensor? = nil
        var targetTensor: MPSGraphTensor? = nil
        var logitsTensor: MPSGraphTensor? = nil
        var lossTensor: MPSGraphTensor? = nil
        var gradientsOutput: [MPSGraphTensor: MPSGraphTensor]? = nil
        var prevAccumulatedGradientPlaceholders: [MPSGraphTensor] = []
        var accumulatedGradientData: [MPSGraphTensorData] = []
        var accumulatedGradientOutputPlaceholders: [MPSGraphTensor] = []
        var targetTensors: [MPSGraphTensor] = []
        
        let tensorQueue = MobilePipeUtils.GlobalTensorQueue
        
        while true {
            let NOF_MICROBATCHES = try await c.receiveUInt8()  // nof microbatches per batch
            if NOF_MICROBATCHES == 0 {
                break
            }
            Task.detached {
                for _ in 0..<NOF_MICROBATCHES {  // input, target
                    let input = try await c.receiveSendableTensor()
                    let target = try await c.receiveSendableTensor()
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

                    logitsTensor = g.ResNet_S2_PreLayer4(graph: C3_PreL4_GRAPH, input: inputTensor!, params: C3_paramPlaceholders, frozenRunningParams: true)

                    lossTensor = g.CrossEntropy(graph: C3_PreL4_GRAPH, input: logitsTensor!, target: targetTensor!, name: "Loss")
                    
                    targetTensors.removeAll()

                    C3_placeholdersToTargetForGradients.append(inputTensor!)
                    gradientsOutput = C3_PreL4_GRAPH.gradients(of: lossTensor!, with: C3_placeholdersToTargetForGradients, name: "Gradients")
                    for i in 0..<C3_placeholdersToTargetForGradients.count {
                        // add all target grads to graph result
                        targetTensors.append(gradientsOutput![C3_placeholdersToTargetForGradients[i]]!)
                    }

                    var i = 0
                    for a in 0..<TRAINABLE_PARAM_IDS.count {
                        if TRAINABLE_PARAM_IDS[a] <= NOF_L2_PARAMS + NOF_L3_PARAMS {
                            continue
                        }
                        let shape = ENV.parameterHandler.getParameterShape(sendableID: TRAINABLE_PARAM_IDS[a])
                        accumulatedGradientData.append(MobilePipeUtils.zerosTensorData(ENV, shape: shape))
                        prevAccumulatedGradientPlaceholders.append(ENV.graphHandler.makePlaceholderFloat32Tensor(graph: C3_PreL4_GRAPH,
                                                                                                                 shape: shape,
                                                                                                                 name: "prevAcumulatedGrad:\(i)"))
                        accumulatedGradientOutputPlaceholders.append(C3_PreL4_GRAPH.addition(prevAccumulatedGradientPlaceholders[i],
                                                                                  gradientsOutput![C3_placeholdersToTargetForGradients[i]]!,
                                                                                  name: "newAccumlatedGrad:\(i)"))  // curSumGrad + newGrad
                        i += 1
                    }
                    targetTensors.append(contentsOf: accumulatedGradientOutputPlaceholders)
                    targetTensors.append(lossTensor!)

                    placeholdersBuilt = true
                }

                // update feeds
                C3_feeds[inputTensor!] = ENV.graphHandler.makeTensorData(input)
                C3_feeds[targetTensor!] = ENV.graphHandler.makeTensorData(target)
                for i in 0..<prevAccumulatedGradientPlaceholders.count {
                    C3_feeds[prevAccumulatedGradientPlaceholders[i]] = accumulatedGradientData[i]
                }

                // run
                let results = C3_PreL4_GRAPH.run(feeds: C3_feeds, targetTensors: targetTensors, targetOperations: nil)

                // send gradient & loss to s1_backward
                c.sendSendableTensor(tensor: DataHandler.toSendableFloat32Tensor(results[gradientsOutput![inputTensor!]!]!))
                c.sendSendableTensor(tensor: DataHandler.toSendableFloat32Tensor(results[lossTensor!]!))

                // update accumulated gradients
                for i in 0..<accumulatedGradientData.count {
                    accumulatedGradientData[i] = results[accumulatedGradientOutputPlaceholders[i]]!
                }
                ENV.TIMEK.end(cat: TimeKeeper.CATS.S2_COMBINED_MICROBATCH)
            }

            ENV.TIMEK.start(cat: TimeKeeper.CATS.OPTIMIZATION_STEP)
            optimizer.step(accumulatedGradientsData: accumulatedGradientData, onlyUseParamsGreaterThan: NOF_L2_PARAMS + NOF_L3_PARAMS)
            // zero grad
            for i in 0..<accumulatedGradientData.count {
                accumulatedGradientData[i] = MobilePipeUtils.zerosTensorData(ENV, shape: accumulatedGradientData[i].shape)
            }

            // update G3_feeds[all G3_placeholders] = all updated weights
            for i in 0..<TRAINABLE_PARAM_IDS.count {
                if TRAINABLE_PARAM_IDS[i] <= NOF_L2_PARAMS + NOF_L3_PARAMS {
                    continue
                }
                C3_feeds[C3_paramPlaceholders[TRAINABLE_PARAM_IDS[i] - 1 - NOF_L2_PARAMS - NOF_L3_PARAMS]] = p.getParameterTensorData(sendableID: TRAINABLE_PARAM_IDS[i])
            }
            ENV.TIMEK.end(cat: TimeKeeper.CATS.OPTIMIZATION_STEP)
        }
        
        // sync final
        for i in 0..<TRAINABLE_PARAM_IDS.count {
            if TRAINABLE_PARAM_IDS[i] <= NOF_L2_PARAMS + NOF_L3_PARAMS {
                continue
            }
            c.sendDouble(Double(TRAINABLE_PARAM_IDS[i]))
            c.sendSendableTensor(tensor: DataHandler.toSendableFloat32Tensor(p.getParameterTensorData(sendableID: TRAINABLE_PARAM_IDS[i])))
        }
        c.sendDouble(0)
    }
}


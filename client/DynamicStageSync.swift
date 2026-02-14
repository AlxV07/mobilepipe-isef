class DynamicStageSync {
    static func SYNC_POST_C1(_ ENV: IAEnvironment, optimizer: Optimizer, adam: Bool=false) async throws {
        let NOF_PARAMS = MobilePipeUtils.RESNET34_S2_NOF_PARAMS
        for id in 1..<NOF_PARAMS + 1 {
            if (id - 1) % 5 == 0 || id == NOF_PARAMS - 1 || id == NOF_PARAMS {
                if id <= MobilePipeUtils.RESNET34_L2_PARAMS {  // send all of layer 2 (we're continuing to train L3 and L4)
                    // value, moment, velocity
                    let val = DataHandler.toSendableFloat32Tensor(ENV.parameterHandler.getParameterTensorData(sendableID: id))
                    ENV.commHandler.sendSendableTensor(tensor: val)
                    if adam {
                        let o = optimizer as! AdamOptimizer
                        let moment = DataHandler.toSendableFloat32Tensor(o.momentsData[o.PARAM_ID_TO_IDX[id]!])
                        let vel = DataHandler.toSendableFloat32Tensor(o.velocitiesData[o.PARAM_ID_TO_IDX[id]!])
                        ENV.commHandler.sendSendableTensor(tensor: moment)
                        ENV.commHandler.sendSendableTensor(tensor: vel)
                    }
                } else {
                    break
                }
            }
        }
        print("Finished SYNC_POST_C1")
    }
    
    static func SYNC_POST_C2(_ ENV: IAEnvironment, optimizer: Optimizer, adam: Bool=false) async throws {
        let NOF_PARAMS = MobilePipeUtils.RESNET34_S2_NOF_PARAMS
        for id in MobilePipeUtils.RESNET34_L2_PARAMS + 1..<NOF_PARAMS + 1 {
            if (id - 1) % 5 == 0 || id == NOF_PARAMS - 1 || id == NOF_PARAMS {
                if id <= MobilePipeUtils.RESNET34_L2_PARAMS + MobilePipeUtils.RESNET34_L3_PARAMS {  // send all of layer 3 (we're continuing to train L4)
                    // value, moment, velocity
                    let val = DataHandler.toSendableFloat32Tensor(ENV.parameterHandler.getParameterTensorData(sendableID: id))
                    ENV.commHandler.sendSendableTensor(tensor: val)
                    if adam {
                        let o = optimizer as! AdamOptimizer
                        let moment = DataHandler.toSendableFloat32Tensor(o.momentsData[o.PARAM_ID_TO_IDX[id]!])
                        let vel = DataHandler.toSendableFloat32Tensor(o.velocitiesData[o.PARAM_ID_TO_IDX[id]!])
                        ENV.commHandler.sendSendableTensor(tensor: moment)
                        ENV.commHandler.sendSendableTensor(tensor: vel)
                    }
                } else {
                    break
                }
            }
        }
        print("Finished SYNC_POST_C2")
    }
}


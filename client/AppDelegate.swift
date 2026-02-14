import UIKit
import Network
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

class AppDelegate: UIResponder, UIApplicationDelegate {
    let ENV = IAEnvironment()
    
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        UIApplication.shared.isIdleTimerDisabled = true
        ENV.log("--- Starting mobilepipe-app...")
        ENV.commHandler.createListener()
        ENV.commHandler.setNewConnectionHandler({ connection in
            self.ENV.log("=== Host Connection received.")
            connection.start(queue: .main)
            self.startOperationLoop(connection)
        })
        ENV.commHandler.startListener()
        ENV.log("--- Listening on iOS port \"\(ENV.commHandler.port)\" with connection type \"\(ENV.commHandler.type)\"...")
        return true;
    }
    
    func startOperationLoop(_ connection: NWConnection) {
        ENV.commHandler.setConnection(connection)
        self.handleOperationIteration()
    }
    
    func handleOperationIteration() {
        Task {
            ENV.log("--- Awaiting opID...")
            let opID = try await ENV.commHandler.receiveOpID()
            ENV.log("=== Received opID: \(opID)")
            do {
                switch opID {
                case OperationID.parameter:
                    ENV.log("--- Beginning parameter store operation...", skip: true)
                    try await parameterOperation()
                case OperationID.input:
                    ENV.log("--- Beginning input operation...")
                    try await inputOperation()
                default:
                    fatalError("*** Unknown Operation ID: \(opID)")
                }
            } catch {
                ENV.log("*** Error while handling operation (ID=\(opID)): \(error)")
            }
            // Recursive:
            handleOperationIteration()
        }
    }
    
    func parameterOperation() async throws {
        let sendableID = try await ENV.commHandler.receiveUInt16()
        let tensor = try await ENV.commHandler.receiveSendableTensor()
        ENV.parameterHandler.storeParameter(sendableID: sendableID, tensor: tensor)
        ENV.log("=== Stored parameter for sendableID: \(sendableID)")
    }
   
    func inputOperation() async throws {
        let inputOpID = try await ENV.commHandler.receiveUInt16()
        ENV.log("=== Received inputOpID: \(inputOpID)")
        switch inputOpID {
        case InputOperationID.MobilePipe_ResNet_Train:
            try await MobilePipeExperiments.MobilePipe_ResNet_Train(ENV)
        default:
            fatalError("*** Unknown Input Operation ID: \(inputOpID)")
        }
    }
}


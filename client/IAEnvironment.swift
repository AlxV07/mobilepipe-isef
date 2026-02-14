import Metal


class IAEnvironment {
    let commHandler: CommHandler
    let graphHandler: GraphHandler
    let parameterHandler: ParameterHandler
    let device: MTLDevice
    let TIMEK: TimeKeeper
    
    init(commHandler: CommHandler, graphHandler: GraphHandler, weightsHandler: ParameterHandler, device: MTLDevice) {
        self.commHandler = commHandler
        self.graphHandler = graphHandler
        self.parameterHandler = weightsHandler
        self.device = device
        self.TIMEK = TimeKeeper()
        //
        self.parameterHandler.setDevice(self.device)
        self.graphHandler.setDevice(self.device)
    }
    
    init() {
        self.commHandler = CommHandler()
        self.graphHandler = GraphHandler()
        self.parameterHandler = ParameterHandler()
        guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device available.") }
        self.device = device
        self.TIMEK = TimeKeeper()
        //
        self.parameterHandler.setDevice(self.device)
        self.graphHandler.setDevice(self.device)
    }
    
    func log(_ i: Any..., skip: Bool = false) {
        for x in i {
            print(x, terminator: " ")
            if (!skip) {
                ConsoleLogger.shared.log("\(x)")
            }
        }
        print()
        // add additional logging functionality here
    }
}


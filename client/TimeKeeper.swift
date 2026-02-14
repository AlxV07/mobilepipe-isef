import Foundation
import Dispatch


class TimeKeeper {
    
    enum CATS: CaseIterable {
        case S2_COMBINED_MICROBATCH
        case WAITING_FOR_INPUT
        case OPTIMIZATION_STEP
        
        static var allCases: [CATS] {
            return [.S2_COMBINED_MICROBATCH, .WAITING_FOR_INPUT, .OPTIMIZATION_STEP]
        }
    }
    
    class TimePair {
        private var time: [Double] = [-1.0, -1.0]
        
        init(start: Double = -1.0, end: Double = -1.0) {
            self.time = [start, end]
        }
        
        func setStart(_ start: Double) {
            self.time[0] = start
        }
        
        func getStart() -> Double {
            return self.time[0]
        }
        
        func setEnd(_ end: Double) {
            self.time[1] = end
        }
        
        func getEnd() -> Double {
            return self.time[1]
        }
        
        func getTime() -> [Double] {
            return self.time
        }
    }
    
    private var times: [CATS: [TimePair]] = [:]
    
    init() {
        for cat in CATS.allCases {
            times[cat] = []
        }
    }
    
    private func _curTime() -> Double {
        return Double(Date().timeIntervalSince1970)
    }
    
    func start(cat: CATS) {
        let timePair = TimePair(start: _curTime())
        times[cat]?.append(timePair)
    }
    
    func end(cat: CATS) {
        if let lastPair = times[cat]?.last {
            lastPair.setEnd(_curTime())
        }
    }
    
    func send_ios_times(_ c: CommHandler) {
        /*
         1. Send the total number of values (remember: nof values, not pairs) for the receiver to expect
         2. For each of these time pairs, send each's start and end value, one after the other, in this order:
            - the first WAITING_FOR_INPUT time pair
            - each of the S2_COMBINED_MICROBATCH time pairs
            - the first OPTIMIZATION_STEP time pair
         To send a value, uses `c.sendDouble(Double)`
         */

        var totalValues = 0

        if let waitingInputTimes = times[.WAITING_FOR_INPUT] {
            totalValues += waitingInputTimes.count * 2 // start and end of each pair
        }

        if let s2Times = times[.S2_COMBINED_MICROBATCH] {
            totalValues += s2Times.count * 2 // start and end of each pair
        }

        if let optStepTimes = times[.OPTIMIZATION_STEP], !optStepTimes.isEmpty {
            totalValues += 2 // start and end of first pair
        }

        c.sendDouble(Double(totalValues))

        if let waitingInputTimes = times[.WAITING_FOR_INPUT] {
            for pair in waitingInputTimes {
                c.sendDouble(pair.getStart())
                c.sendDouble(pair.getEnd())
            }
        }

        if let s2Times = times[.S2_COMBINED_MICROBATCH] {
            for pair in s2Times {
                c.sendDouble(pair.getStart())
                c.sendDouble(pair.getEnd())
            }
        }

        if let optStepTimes = times[.OPTIMIZATION_STEP], !optStepTimes.isEmpty {
            let firstPair = optStepTimes[0]
            c.sendDouble(firstPair.getStart())
            c.sendDouble(firstPair.getEnd())
        }
    }
    
    func clear() {
        times.removeAll()
        for cat in CATS.allCases {
            times[cat] = []
        }
    }
}


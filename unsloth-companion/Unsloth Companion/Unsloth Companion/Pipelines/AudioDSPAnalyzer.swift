import Accelerate
import Foundation

struct AudioDSPResult: Codable, Sendable, Equatable {
    let durationSeconds: Double
    let sampleRate: Double
    let rmsDBFS: Double
    let peakDBFS: Double
    let truePeakDBTP: Double
    let integratedLUFS: Double
    let estimatedBPM: Double?
    let estimatedPitchHz: Double?
    let spectralCentroidHz: Double
    let transientCount: Int
    let clippedSampleCount: Int
}

enum AudioDSPError: LocalizedError {
    case emptyAudio, unsupportedSampleRate

    var errorDescription: String? {
        switch self {
        case .emptyAudio: return String(localized: "The audio stream is empty.")
        case .unsupportedSampleRate: return String(localized: "The audio sample rate is invalid.")
        }
    }
}

enum AudioDSPAnalyzer {
    static func analyze(samples: [Float], sampleRate: Double) throws -> AudioDSPResult {
        guard !samples.isEmpty else { throw AudioDSPError.emptyAudio }
        guard sampleRate > 0 else { throw AudioDSPError.unsupportedSampleRate }

        var rms: Float = 0
        vDSP_rmsqv(samples, 1, &rms, vDSP_Length(samples.count))
        var peak: Float = 0
        vDSP_maxmgv(samples, 1, &peak, vDSP_Length(samples.count))
        let truePeak = sincTruePeak(samples)
        let energies = shortTermEnergies(samples: samples, sampleRate: sampleRate)
        return AudioDSPResult(
            durationSeconds: Double(samples.count) / sampleRate,
            sampleRate: sampleRate,
            rmsDBFS: amplitudeDB(Double(rms)),
            peakDBFS: amplitudeDB(Double(peak)),
            truePeakDBTP: amplitudeDB(Double(truePeak)),
            integratedLUFS: integratedLUFS(samples: samples, sampleRate: sampleRate),
            estimatedBPM: estimateBPM(energies: energies, sampleRate: sampleRate),
            estimatedPitchHz: estimatePitchYIN(samples: samples, sampleRate: sampleRate),
            spectralCentroidHz: spectralCentroid(samples: samples, sampleRate: sampleRate),
            transientCount: transientCount(energies),
            clippedSampleCount: samples.reduce(0) { $0 + (abs($1) >= 0.999 ? 1 : 0) }
        )
    }

    private static func amplitudeDB(_ value: Double) -> Double {
        20 * log10(max(value, 1e-12))
    }

    private static func sincTruePeak(_ samples: [Float]) -> Float {
        guard samples.count > 1 else { return abs(samples[0]) }
        let taps = 12
        var maximum: Float = 0
        for index in samples.indices {
            maximum = max(maximum, abs(samples[index]))
            guard index + 1 < samples.count else { continue }
            for phase in 1..<4 {
                let fraction = Double(phase) / 4
                var value = 0.0
                var weight = 0.0
                for tap in -taps...taps {
                    let source = index + tap
                    guard samples.indices.contains(source) else { continue }
                    let x = Double(tap) - fraction
                    let sinc = x == 0 ? 1 : sin(.pi * x) / (.pi * x)
                    let window = 0.5 + 0.5 * cos(.pi * x / Double(taps + 1))
                    let coefficient = sinc * window
                    value += Double(samples[source]) * coefficient
                    weight += coefficient
                }
                if weight != 0 { maximum = max(maximum, Float(abs(value / weight))) }
            }
        }
        return maximum
    }

    private static func integratedLUFS(samples: [Float], sampleRate: Double) -> Double {
        let normalized = sampleRate == 48_000 ? samples : resampleLinear(samples, from: sampleRate, to: 48_000)
        let shelf = biquad(normalized, b: [1.53512485958697, -2.69169618940638, 1.19839281085285], a: [1, -1.69065929318241, 0.73248077421585])
        let weighted = biquad(shelf, b: [1, -2, 1], a: [1, -1.99004745483398, 0.99007225036621])
        let window = 19_200
        let hop = 4_800
        guard weighted.count >= window else {
            let power = weighted.reduce(0.0) { $0 + Double($1 * $1) } / Double(max(1, weighted.count))
            return -0.691 + 10 * log10(max(power, 1e-12))
        }
        var powers: [Double] = []
        var offset = 0
        while offset + window <= weighted.count {
            var meanSquare: Float = 0
            vDSP_measqv(Array(weighted[offset..<(offset + window)]), 1, &meanSquare, vDSP_Length(window))
            let power = Double(meanSquare)
            if -0.691 + 10 * log10(max(power, 1e-12)) > -70 { powers.append(power) }
            offset += hop
        }
        guard !powers.isEmpty else { return -70 }
        let ungatedPower = powers.reduce(0, +) / Double(powers.count)
        let relativeGate = -0.691 + 10 * log10(max(ungatedPower, 1e-12)) - 10
        let gated = powers.filter { -0.691 + 10 * log10(max($0, 1e-12)) >= relativeGate }
        let integratedPower = gated.reduce(0, +) / Double(max(1, gated.count))
        return -0.691 + 10 * log10(max(integratedPower, 1e-12))
    }

    private static func biquad(_ input: [Float], b: [Double], a: [Double]) -> [Float] {
        var output = [Float](repeating: 0, count: input.count)
        var x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0
        for index in input.indices {
            let x = Double(input[index])
            let y = b[0] * x + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            output[index] = Float(y)
            x2 = x1; x1 = x; y2 = y1; y1 = y
        }
        return output
    }

    private static func resampleLinear(_ samples: [Float], from sourceRate: Double, to targetRate: Double) -> [Float] {
        let count = max(1, Int(Double(samples.count) * targetRate / sourceRate))
        return (0..<count).map { index in
            let position = Double(index) * sourceRate / targetRate
            let low = min(samples.count - 1, Int(position))
            let high = min(samples.count - 1, low + 1)
            let mix = Float(position - Double(low))
            return samples[low] * (1 - mix) + samples[high] * mix
        }
    }

    private static func shortTermEnergies(samples: [Float], sampleRate: Double) -> [Double] {
        let frame = max(128, Int(sampleRate * 0.021333))
        let hop = max(64, frame / 2)
        guard samples.count >= frame else { return [] }
        var result: [Double] = []
        var offset = 0
        while offset + frame <= samples.count {
            var meanSquare: Float = 0
            vDSP_measqv(Array(samples[offset..<(offset + frame)]), 1, &meanSquare, vDSP_Length(frame))
            result.append(Double(meanSquare))
            offset += hop
        }
        return result
    }

    private static func estimateBPM(energies: [Double], sampleRate: Double) -> Double? {
        guard energies.count > 32 else { return nil }
        let hopSeconds = 0.0106665
        let onset = zip(energies.dropFirst(), energies).map { max(0, $0 - $1) }
        let minLag = max(1, Int(60 / 200 / hopSeconds))
        let maxLag = min(onset.count - 1, Int(60 / 60 / hopSeconds))
        guard minLag < maxLag else { return nil }
        var bestLag = minLag
        var best = -Double.infinity
        for lag in minLag...maxLag {
            var score = 0.0
            for index in lag..<onset.count { score += onset[index] * onset[index - lag] }
            if score > best { best = score; bestLag = lag }
        }
        guard best > 0 else { return nil }
        return 60 / (Double(bestLag) * hopSeconds)
    }

    private static func estimatePitchYIN(samples: [Float], sampleRate: Double) -> Double? {
        let frameCount = min(samples.count, 8_192)
        guard frameCount >= 1_024 else { return nil }
        let frame = Array(samples.prefix(frameCount))
        let minLag = max(2, Int(sampleRate / 1_500))
        let maxLag = min(frameCount / 2, Int(sampleRate / 50))
        guard minLag < maxLag else { return nil }
        var difference = [Double](repeating: 0, count: maxLag + 1)
        for lag in 1...maxLag {
            var sum = 0.0
            for index in 0..<(frameCount - lag) {
                let delta = Double(frame[index] - frame[index + lag])
                sum += delta * delta
            }
            difference[lag] = sum
        }
        var cumulative = 0.0
        var normalized = difference
        for lag in 1...maxLag {
            cumulative += difference[lag]
            normalized[lag] = difference[lag] * Double(lag) / max(cumulative, 1e-12)
        }
        guard let lag = (minLag..<maxLag).first(where: { normalized[$0] < 0.15 && normalized[$0] <= normalized[$0 + 1] }) else { return nil }
        return sampleRate / Double(lag)
    }

    private static func spectralCentroid(samples: [Float], sampleRate: Double) -> Double {
        let n = min(4_096, 1 << Int(floor(log2(Double(samples.count)))))
        guard n >= 64, let setup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(n), .FORWARD) else { return 0 }
        defer { vDSP_DFT_DestroySetup(setup) }
        var realInput = [Float](repeating: 0, count: n / 2)
        var imaginaryInput = [Float](repeating: 0, count: n / 2)
        for index in 0..<(n / 2) {
            realInput[index] = samples[index * 2]
            imaginaryInput[index] = samples[index * 2 + 1]
        }
        var realOutput = [Float](repeating: 0, count: n / 2)
        var imaginaryOutput = [Float](repeating: 0, count: n / 2)
        vDSP_DFT_Execute(setup, realInput, imaginaryInput, &realOutput, &imaginaryOutput)
        var numerator = 0.0, denominator = 0.0
        for index in realOutput.indices {
            let magnitude = hypot(Double(realOutput[index]), Double(imaginaryOutput[index]))
            numerator += Double(index) * sampleRate / Double(n) * magnitude
            denominator += magnitude
        }
        return denominator > 0 ? numerator / denominator : 0
    }

    private static func transientCount(_ energies: [Double]) -> Int {
        guard energies.count > 2 else { return 0 }
        let differences = zip(energies.dropFirst(), energies).map { max(0, $0 - $1) }
        let mean = differences.reduce(0, +) / Double(differences.count)
        let variance = differences.reduce(0) { $0 + pow($1 - mean, 2) } / Double(differences.count)
        let threshold = mean + 2 * sqrt(variance)
        return differences.filter { $0 > threshold }.count
    }
}

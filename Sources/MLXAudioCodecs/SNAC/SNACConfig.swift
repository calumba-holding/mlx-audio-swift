//
//  SNACConfig.swift
//  MLXAudio
//
//  Created by Ben Harraway on 14/05/2025.
//
import Foundation

public struct SNACConfig {
    public let samplingRate: Int
    public let encoderDim: Int
    public let encoderRates: [Int]
    public let decoderDim: Int
    public let decoderRates: [Int]
    public let attnWindowSize: Int?
    public let codebookSize: Int
    public let codebookDim: Int
    public let vqStrides: [Int]
    public let noise: Bool
    public let depthwise: Bool
    public let latentDim: Int

    public init(
        samplingRate: Int = 24000,
        encoderDim: Int = 48,
        encoderRates: [Int] = [2, 4, 8, 8],
        decoderDim: Int = 1024,
        decoderRates: [Int] = [8, 8, 4, 2],
        attnWindowSize: Int? = nil,
        codebookSize: Int = 4096,
        codebookDim: Int = 8,
        vqStrides: [Int] = [4, 2, 1],
        noise: Bool = true,
        depthwise: Bool = true,
        latentDim: Int? = nil
    ) {
        self.samplingRate = samplingRate
        self.encoderDim = encoderDim
        self.encoderRates = encoderRates
        self.decoderDim = decoderDim
        self.decoderRates = decoderRates
        self.attnWindowSize = attnWindowSize
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self.vqStrides = vqStrides
        self.noise = noise
        self.depthwise = depthwise
        
        // Calculate latentDim if not provided
        if let latentDim = latentDim {
            self.latentDim = latentDim
        } else {
            self.latentDim = encoderDim * Int(pow(2.0, Double(encoderRates.count)))
        }
    }
}

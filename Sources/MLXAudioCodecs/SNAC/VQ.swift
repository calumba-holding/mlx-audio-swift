//
//  VQ.swift
//  MLXAudio
//
//  Created by Prince Canuma on 29/12/25.
//

import Foundation
import MLX
import MLXNN

// MARK: - Helper Functions

func normalize(_ x: MLXArray, p: Float = 2.0, dim: Int = 1, eps: Float = 1e-12) -> MLXArray {
    let norm = pow(
        sum(pow(abs(x), MLXArray(p)), axis: dim, keepDims: true),
        MLXArray(1.0 / p)
    )
    return x / maximum(norm, MLXArray(eps))
}

// MARK: - Vector Quantize

public class VectorQuantize: Module {
    let codebookSize: Int
    let codebookDim: Int
    let stride: Int
    let inProj: WNConv1d
    let outProj: WNConv1d
    let codebook: Embedding
    
    public init(
        inputDim: Int,
        codebookSize: Int,
        codebookDim: Int,
        stride: Int = 1
    ) {
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self.stride = stride
        
        self.inProj = WNConv1d(inChannels: inputDim, outChannels: codebookDim, kernelSize: 1)
        self.outProj = WNConv1d(inChannels: codebookDim, outChannels: inputDim, kernelSize: 1)
        self.codebook = Embedding(embeddingCount: codebookSize, dimensions: codebookDim)
    }
    
    public func callAsFunction(_ z: MLXArray) -> (MLXArray, MLXArray) {
        var z = z.swappedAxes(1, 2)
        
        if stride > 1 {
            let kernelSize = stride
            let stride = self.stride
            let kernel = ones([z.shape[2], kernelSize, 1]) / Float(kernelSize)
            z = conv1d(z, kernel, stride: stride, padding: 0, groups: z.shape[2])
        }
        
        // Factorized codes - Project input into low-dimensional space
        let zE = inProj(z).swappedAxes(1, 2)  // z_e : (B x D x T)
        let (zQ, indices) = decodeLatents(zE)
        
        // Straight-through estimator: z_e + stop_gradient(z_q - z_e)
        var zQFinal = zE + stopGradient(zQ - zE)
        
        zQFinal = outProj(zQFinal.swappedAxes(1, 2)).swappedAxes(1, 2)
        
        if stride > 1 {
            // Implement repeat_interleave
            var shape = zQFinal.shape
            shape[shape.count - 1] *= stride
            var expanded = zeros(shape)
            
            // Fill the expanded tensor with repeated values
            for i in 0..<stride {
                expanded[.ellipsis, i.stride(stride) as! MLXArrayIndex] = zQFinal
            }
            
            zQFinal = expanded
        }
        
        return (zQFinal, indices)
    }
    
    public func embedCode(_ embedId: MLXArray) -> MLXArray {
        return codebook.weight[embedId]
    }
    
    public func decodeCode(_ embedId: MLXArray) -> MLXArray {
        return embedCode(embedId).swappedAxes(1, 2)
    }
    
    public func decodeLatents(_ latents: MLXArray) -> (MLXArray, MLXArray) {
        // Rearrange: "b d t -> (b t) d"
        let B = latents.shape[0]
        let D = latents.shape[1]
        let T = latents.shape[2]
        var encodings = latents.transposed(0, 2, 1)  // [B, T, D]
        encodings = encodings.reshaped([B * T, D])
        
        let codebookWeights = codebook.weight  // codebook: (N x D)
        
        let encodingsNorm = normalize(encodings)
        let codebookNorm = normalize(codebookWeights)
        
        let dist = sum(pow(encodingsNorm, MLXArray(2)), axis: 1, keepDims: true)
            - 2 * matmul(encodingsNorm, codebookNorm.T)
            + sum(pow(codebookNorm, MLXArray(2)), axis: 1, keepDims: true).T
        
        let minDist = argMax(-dist, axis: 1)
        
        // Rearrange: "(b t) -> b t"
        let indices = minDist.reshaped([B, T])
        let zQ = decodeCode(indices)
        
        return (zQ, indices)
    }
}

// MARK: - Residual Vector Quantize

public class ResidualVectorQuantize: Module {
    let nCodebooks: Int
    let codebookDim: Int
    let codebookSize: Int
    let quantizers: [VectorQuantize]
    
    public init(
        inputDim: Int = 512,
        codebookSize: Int = 1024,
        codebookDim: Int = 8,
        vqStrides: [Int] = [1, 1, 1, 1]
    ) {
        self.nCodebooks = vqStrides.count
        self.codebookDim = codebookDim
        self.codebookSize = codebookSize
        self.quantizers = vqStrides.map { stride in
            VectorQuantize(
                inputDim: inputDim,
                codebookSize: codebookSize,
                codebookDim: codebookDim,
                stride: stride
            )
        }
    }
    
    public func callAsFunction(_ z: MLXArray) -> (MLXArray, [MLXArray]) {
        var zQ = zeros(z.shape)
        var residual = z
        var codes: [MLXArray] = []
        
        for quantizer in quantizers {
            let (zQI, indicesI) = quantizer(residual)
            zQ = zQ + zQI
            residual = residual - zQI
            codes.append(indicesI)
        }
        
        return (zQ, codes)
    }
    
    public func fromCodes(_ codes: [MLXArray]) -> MLXArray {
        var zQ = MLXArray(0.0)
        
        for i in 0..<nCodebooks {
            let zPI = quantizers[i].decodeCode(codes[i])
            var zQI = quantizers[i].outProj(zPI.swappedAxes(1, 2)).swappedAxes(1, 2)
            
            // Handle repeat_interleave for stride > 1
            if quantizers[i].stride > 1 {
                let stride = quantizers[i].stride
                var shape = zQI.shape
                shape[shape.count - 1] *= stride
                var expanded = zeros(shape)
                
                for j in 0..<stride {
                    expanded[.ellipsis, j.stride(stride) as! MLXArrayIndex] = zQI
                }
                
                zQI = expanded
            }
            
            zQ = zQ + zQI
        }
        
        return zQ
    }
}

// MARK: - Helper Extensions

extension Int {
    func stride(_ stride: Int) -> StrideThrough<Int> {
        return Swift.stride(from: self, through: Int.max, by: stride)
    }
}

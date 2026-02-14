import MetalPerformanceShadersGraph


class GraphHandler {
    var device: MTLDevice?
    var commandQueue: MTLCommandQueue?
    
    init() {}
    
    // ======= GraphHandler Attributes =======
    
    func setDevice(_ device: MTLDevice) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
    }
    
    func getCommandQueue() -> MTLCommandQueue {
        return self.commandQueue!
    }
    
    // ======= Tensor Data Handling =======
    
    func makeTensorBuffer(_ tensor: SendableTensor) -> MTLBuffer {
        guard let buffer = device!.makeBuffer(
            bytes: (tensor.tensorData as NSData).bytes,
            length: tensor.nofBytes,
            options: .storageModeShared)
        else {
            fatalError("Failed to make tensor buffer.")
        }
        return buffer;
    }
    
    func makeTensorData(_ tensor: SendableTensor) -> MPSGraphTensorData {
        return MPSGraphTensorData(self.makeTensorBuffer(tensor), shape: tensor.dims, dataType: tensor.dataType)
    }
    
    func makePlaceholderFloat32Tensor(graph: MPSGraph, shape: [NSNumber], name: String) -> MPSGraphTensor {
        return graph.placeholder(shape: shape, dataType: .float32, name: name)
    }
    
    // ======= General Methods =======
    
    func Mult(graph: MPSGraph, inp1: MPSGraphTensor, inp2: MPSGraphTensor, name: String) -> MPSGraphTensor {
        // TODO: remove this method, just use `graph.multiplication` directly for simplicity
        // inp1 = input1, inp2 = input2
        return graph.multiplication(inp1, inp2, name: "Mult:" + name)
    }
    
    func Linear(graph: MPSGraph, input: MPSGraphTensor, weight: MPSGraphTensor, bias: MPSGraphTensor? = nil, name: String) -> MPSGraphTensor {
        // inp1 = input, inp2 = weight
        var out = graph.matrixMultiplication(primary: input, secondary: weight, name: "Linear:" + name)
        if let bias = bias {
            out = graph.addition(out, bias, name: "Linear:" + name + "+AddBias")
        }
        return out
    }
    
    func SiLU(graph: MPSGraph, input: MPSGraphTensor, name: String) -> MPSGraphTensor {
        // inp1 = input
        return graph.multiplication(input, graph.sigmoid(with: input, name: "SiLU:" + name + "inner_sigmoid"), name: "SiLU:" + name)
    }
    
    func GELU(graph: MPSGraph, input: MPSGraphTensor) -> MPSGraphTensor {
        let dataType = input.dataType
        let coeffA = graph.constant(0.044715, shape: [], dataType: dataType)
        let coeffB = graph.constant(sqrt(2.0 / .pi), shape: [], dataType: dataType)
        let one = graph.constant(1.0, shape: [], dataType: dataType)
        let half = graph.constant(0.5, shape: [], dataType: dataType)
        let pow = graph.constant(3.0, shape: [], dataType: dataType)
        let xCubed = graph.power(input, pow, name: "pow")  // TODO: Assign real names
        let inner = graph.addition(input, graph.multiplication(coeffA, xCubed, name: nil), name: nil)
        let scaled = graph.multiplication(coeffB, inner, name: nil)
        let tanhOut = graph.tanh(with: scaled, name: nil)
        let onePlusTanh = graph.addition(one, tanhOut, name: nil)
        let gelu = graph.multiplication(input, graph.multiplication(half, onePlusTanh, name: nil), name: "GeLU")
        return gelu
    }
    
    func MatMul4D_ByHand(  
        _ graph: MPSGraph,
        _ A: MPSGraphTensor,
        _ B: MPSGraphTensor,
        name: String? = "MatMul4D_ByHand"
    ) -> MPSGraphTensor {
//        return graph.matrixMultiplication(primary: A, secondary: B, name: name)
        // 1) add a trailing axis to A:  [N, H, M, K]   -> [N, H, M, K, 1]
        let A_exp = graph.expandDims(
            A,
            axes: [NSNumber(value: 4)],          // append at the end
            name: name.map { "\($0)/A_exp" }
        )
        // 2) add an axis before K on B: [N, H, K, P] -> [N, H, 1, K, P]
        let B_exp = graph.expandDims(
            B,
            axes: [NSNumber(value: 2)],          // insert at index 2
            name: name.map { "\($0)/B_exp" }
        )
        // 3) broadcast multiply: [N, H, M, K, 1] * [N, H, 1, K, P] -> [N, H, M, K, P]
        let prod = graph.multiplication(
            A_exp,
            B_exp,
            name: name.map { "\($0)/prod" }
        )
        // prod: [N, H, M, K, P]
        let sumK = graph.reductionSum(
            with: prod,
            axes: [NSNumber(value: 3)],              // reduce over K
            name: name.map { "\($0)/sumK" }
        )
        // remove the kept dim -> [N, H, M, P]
        let C = graph.squeeze(
            sumK,
            axes: [NSNumber(value: 3)],
            name: name.map { "\($0)/squeezeK" }
        )
        return C
    }
    
    func LayerNorm(graph: MPSGraph,
                   normalizedShape: [NSNumber],
                   input: MPSGraphTensor,
                   weight: MPSGraphTensor,
                   bias: MPSGraphTensor,
                   epsilon: Double = 1e-5,
                   name: String) -> MPSGraphTensor {
        let rank = input.shape!.count
        let axes = Array((rank - normalizedShape.count)..<rank).map { NSNumber(value: $0) }
        // 1. mean
        let mean = graph.mean(of: input, axes: axes, name: "LayerNorm:" + name + ":mean")  // TODO: Assign real names
        // 2. variance
        let diff = graph.subtraction(input, mean, name: "b")
        let sq = graph.square(with: diff, name: "c")
        let var_ = graph.mean(of: sq, axes: axes, name: "d")
        // 3. normalize: (x - mean) / sqrt(var + eps)
        let epsTensor = graph.constant(epsilon, shape: [], dataType: input.dataType)
        let denom = graph.squareRoot(with: graph.addition(var_, epsTensor, name: "e"), name: "f")
        let normed = graph.division(diff, denom, name: "g")
        // 4. apply scale (weight) and shift (bias)
        var output = normed
        output = graph.multiplication(output, weight, name: "h")
        output = graph.addition(output, bias, name: "i")
        return output
    }
    
    func Dropout(graph: MPSGraph, input: MPSGraphTensor, rate: Double = 0.1) -> MPSGraphTensor {
        // fix: MPS built in dropout does `x / p` instead of `x / (1 - p)`
        let dropout = graph.dropout(graph.constant(rate, shape: input.shape!, dataType: input.dataType), rate: rate, name: "dropout")  // creates dropout-ed `1` mask
        return graph.multiplication(dropout, input, name: "dropout_mul")
    }
    
    func Conv2D(graph: MPSGraph, input: MPSGraphTensor, weight: MPSGraphTensor, bias: MPSGraphTensor? = nil, name: String,
                strideX: Int = 1,
                strideY: Int = 1,
                paddingX: Int = 1,
                paddingY: Int = 1,
    ) -> MPSGraphTensor {
        let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: strideX,
            strideInY: strideX,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingStyle: MPSGraphPaddingStyle.explicit,
            dataLayout: MPSGraphTensorNamedDataLayout.NCHW,
            weightsLayout: MPSGraphTensorNamedDataLayout.OIHW
        )!
        descriptor.setExplicitPaddingWithPaddingLeft(paddingX, paddingRight: paddingX, paddingTop: paddingY, paddingBottom: paddingY)
        var out = graph.convolution2D(input, weights: weight, descriptor: descriptor, name: "Conv2D:" + name)
        if let b = bias {
            out = graph.addition(out, b, name: "Conv2D:" + name + ":bias")
        }
        return out
    }
    
    func BatchNorm2D(
        graph: MPSGraph,
        input: MPSGraphTensor,
        weight: MPSGraphTensor?,
        bias: MPSGraphTensor?,      
        epsilon: Double = 1e-5,
        name: String,
        frozenRunningParams: Bool = true,
        runningMean: MPSGraphTensor,  // TODO: use updated weights in training, update runningMean
        runningVar: MPSGraphTensor,
    ) -> MPSGraphTensor{
        let axes = [NSNumber(value: 0), 2, 3]
        let meanTensor = (!frozenRunningParams ? graph.mean(of: input, axes: axes, name: "BatchNorm2D:" + name + ":mean") : graph.reshape(runningMean,
                                                                                                                              shape: [1, runningMean.shape![0], 1, 1],
                                                                                                                              name: name + "reshapedRunningMean"))
        let centered = graph.subtraction(input, meanTensor, name: "BatchNorm2D:" + name + ":centered")
        let varTensor = (!frozenRunningParams ? graph.mean(
            of:graph.square(with: centered, name: "BatchNorm2D:" + name + ":squared"),
            axes: axes,
            name: "BatchNorm2D:" + name + ":variance") : graph.reshape(runningVar, shape: [1, runningVar.shape![0], 1, 1], name: name + "reshapedRunningVar"))
        let invStd = graph.reciprocalSquareRoot(
            graph.addition(varTensor, graph.constant(epsilon, dataType: input.dataType), name: "BatchNorm2D:" + name + ":varSquared"),
            name: "BatchNorm2D:" + name + ":invStd")
        var normed = graph.multiplication(centered, invStd, name: "BatchNorm2D:" + name + ":normed")
        if let weight = weight {
            let reshapedWeight = graph.reshape(weight, shape: [1, weight.shape![0], 1, 1], name: "BatchNorm2D:" + name + "reshapeWeight")  // not auto-broadcasted
            normed = graph.multiplication(normed, reshapedWeight, name: "BatchNorm2D:" + name + ":weighted")
        }
        if let bias = bias {
            let reshapedBias = graph.reshape(bias, shape: [1, bias.shape![0], 1, 1], name: "BatchNorm2D:" + name + "reshapeBias")
            normed = graph.addition(normed, reshapedBias, name: "BatchNorm2D:" + name + ":biased")
        }
        return normed
    }
    
    func ReLU(graph: MPSGraph, input: MPSGraphTensor, name: String) -> MPSGraphTensor {
        return graph.reLU(with: input, name: "ReLU:" + name)
    }
    
    func MaxPool2D(graph: MPSGraph, input: MPSGraphTensor, name: String,
                   kernel_size: Int = 3,
                   stride: Int = 2,
                   padding: Int = 1,
                   dilation: Int = 1,
    ) -> MPSGraphTensor {
        let mpD = MPSGraphPooling2DOpDescriptor(
            kernelWidth: kernel_size, kernelHeight: kernel_size,
            strideInX: stride, strideInY: stride,
            dilationRateInX: dilation, dilationRateInY: dilation,
            paddingLeft: padding, paddingRight: padding, paddingTop: padding, paddingBottom: padding,
            paddingStyle: MPSGraphPaddingStyle.explicit,
            dataLayout: MPSGraphTensorNamedDataLayout.NCHW,
        )!
        return graph.maxPooling2D(withSourceTensor: input, descriptor: mpD, name: name)
    }
    
    func AdaptiveAvgPool2D1x1(graph: MPSGraph, input: MPSGraphTensor, name: String) -> MPSGraphTensor {
        /*
         AdaptiveAvgPool((1, 1)) == global average pooling
         */
        let result = graph.mean(of: input, axes: [2, 3] as [NSNumber], name: name + ":mean")
        // reshape to [N, C, 1, 1]
        return graph.reshape(result, shape: [input.shape![0], input.shape![1], 1, 1] as [NSNumber], name: name)
    }
    
    func CrossEntropy(graph: MPSGraph, input: MPSGraphTensor, target: MPSGraphTensor, name: String, isGPT: Bool = false) -> MPSGraphTensor {
//        let depth = isGPT ?
//              50257  // GPT2 vocab size
//            : 1000   // ResNet nof classes
        let depth = 4  // TINY TINY_IMAGENET
//        let depth = 10  // STRESS TINY_IMAGENET
        let oneHot = graph.oneHot(withIndicesTensor: target, depth: depth, name: name + ":oneHot")
        return graph.softMaxCrossEntropy(input,
                                         labels: oneHot,
                                         axis: 1,
                                         reuctionType: MPSGraphLossReductionType.mean,
                                         name: name)
    }
    
    func ForCausalLMLoss(graph: MPSGraph, logitsTensor: MPSGraphTensor, labelsTensor: MPSGraphTensor, vocab_size: Int) -> MPSGraphTensor {
//        let leftPadding = [NSNumber(value: 0), NSNumber(value: 0)]
//        let rightPadding = [NSNumber(value: 0), NSNumber(value: 1)]
//        let shiftLabelsTensor = graph.padTensor(
//            labelsTensor,
//            with: MPSGraphPaddingMode.constant,
//            leftPadding: leftPadding,
//            rightPadding: rightPadding,
//            constantValue: -100,
//            name: "ForCausalLMLoss:pad_labels"
//        )
        let flatLogits = graph.reshape(logitsTensor, shape: [-1, NSNumber(value: vocab_size)], name: "flatten_logits")
//        let flatLabels = graph.reshape(shiftLabelsTensor, shape: [-1], name: "flatten_labels")
        let flatLabels = graph.reshape(labelsTensor, shape: [-1], name: "flatten_labels")
        print(logitsTensor.shape!)
        print(labelsTensor.shape!)
        print(flatLogits.shape!)
        print(flatLabels.shape!)
        return self.CrossEntropy(graph: graph, input: flatLogits, target: flatLabels, name: "ForCausalLMLoss:cross_entropy")
    }
    
    // ======= Qwen3 Methods =======
    
    func Qwen3_RMSNorm(graph: MPSGraph, hidden_states: MPSGraphTensor, weight: MPSGraphTensor, epsilon: Double = 1e-6, name: String) -> MPSGraphTensor {
        // inp1 = input (a.k.a. hidden_states), inp2 = weight
        // variance = hidden_states.pow(2).mean(-1, keepdim=True)
        let squared = graph.square(with: hidden_states, name: "RMSNorm:" + name + "square")
        let mean = graph.mean(of: squared, axes: [NSNumber(value: hidden_states.shape!.count - 1)], name: "RMSNorm:" + name + "mean")
        // normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        let varianceEps = graph.addition(mean, graph.constant(epsilon, dataType: .float32), name: "RMSNorm:" + name + "variance_plus_epsilon")
        let invRMS = graph.reciprocalSquareRoot(varianceEps, name: "RMSNorm:" + name + "inv_rms")
        let normalized = graph.multiplication(hidden_states, invRMS, name: "RMSNorm:" + name + "normalized")
        // return self.weight * normalized
        let output = graph.multiplication(normalized, weight, name: "RMSNorm:" + name + "output")
        return output
    }
    
    func Qwen3_MLP(graph: MPSGraph,
                   _ hidden_states: MPSGraphTensor, _ linear1: MPSGraphTensor, _ linear2: MPSGraphTensor, _ linear3: MPSGraphTensor, name: String) -> MPSGraphTensor {
        // return = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        let gateTensor = self.Linear(graph: graph, input: hidden_states, weight: linear1, name: "GateLinear")
        let upTensor = self.Linear(graph: graph, input: hidden_states, weight: linear2, name: "UpLinear")
        let siLUTensor = self.SiLU(graph: graph, input: gateTensor, name: "SiLUTensor")
        let multTensor = self.Mult(graph: graph, inp1: siLUTensor, inp2: upTensor, name: "MultTensor")
        let downTensor = self.Linear(graph: graph, input: multTensor, weight: linear3, name: "UpLinear")
        return downTensor
    }
    
    func Qwen3_Attention(graph: MPSGraph,
                         hidden_states_tensor: MPSGraphTensor,
                         pos_cos: MPSGraphTensor,
                         pos_sin: MPSGraphTensor,
                         q_proj_weight: MPSGraphTensor,
                         k_proj_weight: MPSGraphTensor,
                         v_proj_weight: MPSGraphTensor,
                         o_proj_weight: MPSGraphTensor,
                         q_norm_weight: MPSGraphTensor,
                         k_norm_weight: MPSGraphTensor,
                         cached_k: MPSGraphTensor? = nil,
                         cached_v: MPSGraphTensor? = nil,
                         //                   c: CommHandler,
                         //    ) -> (attn_output: MPSGraphTensor, cached_k: MPSGraphTensor, cached_v: MPSGraphTensor) {
    ) -> (attn_output: MPSGraphTensor, cached_k: MPSGraphTensor, cached_v: MPSGraphTensor, al: MPSGraphTensor, vl: MPSGraphTensor) {
        //        self.c = c;  // TODO: remove once done debugging
        let head_dim: NSNumber = 128  // TEMP: from default config
        
        // input_shape = hidden_states.shape[:-1]
        let input_shape = hidden_states_tensor.shape!.dropLast()
        // hidden_shape = (*input_shape, -1, self.head_dim)
        let hidden_shape = input_shape + [-1, head_dim] as [NSNumber]
        
        // query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        let qp = self.Linear(graph: graph, input: hidden_states_tensor, weight: q_proj_weight, name: "q_proj-linear")
        let qp_view = graph.reshape(qp, shape: hidden_shape, name: "qp_view")
        let qs = self.Qwen3_RMSNorm(graph: graph, hidden_states: qp_view, weight: q_norm_weight, name: "qs-untransposed")
        var query_states = graph.transposeTensor(qs, dimension: 1, withDimension: 2, name: "query_states")
        
        // key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        let kp = self.Linear(graph: graph, input: hidden_states_tensor, weight: k_proj_weight, name: "k_proj-linear")
        let kp_view = graph.reshape(kp, shape: hidden_shape, name: "kp_view")
        let ks = self.Qwen3_RMSNorm(graph: graph, hidden_states: kp_view, weight: k_norm_weight, name: "ks-untransposed")
        var key_states = graph.transposeTensor(ks, dimension: 1, withDimension: 2, name: "key_states")
        
        // value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        let vp = self.Linear(graph: graph, input: hidden_states_tensor, weight: v_proj_weight, name: "v_proj-linear")
        let vp_view = graph.reshape(vp, shape: hidden_shape, name: "vp_view")
        var value_states = graph.transposeTensor(vp_view, dimension: 1, withDimension: 2, name: "value_states")
        
        // cos, sin = position_embeddings
        // query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        (query_states, key_states) = self.apply_rotary_pos_emb(graph: graph, q: query_states, k: key_states, pe_cos: pos_cos, pe_sin: pos_sin)
        
        // if past_key_value is not None:
        if (cached_k != nil && cached_v != nil
            //            && false
        ) {  // comment `&& false` to let caching run
            //     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            //     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_states = graph.concatTensors([cached_k!, key_states], dimension: cached_k!.shape!.count - 2, name: "cached_k")
            value_states = graph.concatTensors([cached_v!, value_states], dimension: cached_v!.shape!.count - 2, name: "cached_v")
        }
        
        //        MPSGraph.scaledDotProductAttention(
        
        // spda_attention
        var (attn_output, al, vl) = sdpa_attention_forward(graph: graph,
                                                           //        var attn_output = spda_attention_forward(graph: graph,
                                                           num_key_value_groups: 2,  // from qwen default
                                                           query: query_states,
                                                           key_arg: key_states,
                                                           value_arg: value_states,
                                                           attention_mask: nil,
                                                           dropout: 0.0,
                                                           scaling: 1.0 / sqrt(Float32(truncating: head_dim)),
                                                           is_causal_arg: nil) // temp
        
        // attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = graph.reshape(attn_output, shape: input_shape + [-1] as [NSNumber], name: "Attention-attn_output-reshape")
        
        // attn_output: torch.Tensor = self.o_proj(attn_output)
        attn_output = self.Linear(graph: graph, input: attn_output, weight: o_proj_weight, name: "o_proj-linear")
        return (attn_output: attn_output, cached_k: key_states, cached_v: value_states, al: al, vl: vl)
        //        return (attn_output: attn_output, cached_k: key_states, cached_v: value_states)
    }
    
    func Qwen3_Decoder(graph: MPSGraph,
                       hidden_states_tensor: MPSGraphTensor,
                       pos_cos: MPSGraphTensor,
                       pos_sin: MPSGraphTensor,
                       
                       inp_norm_weight: MPSGraphTensor,
                       
                       attn_qproj_weight: MPSGraphTensor,
                       attn_kproj_weight: MPSGraphTensor,
                       attn_vproj_weight: MPSGraphTensor,
                       attn_oproj_weight: MPSGraphTensor,
                       attn_qnorm_weight: MPSGraphTensor,
                       attn_knorm_weight: MPSGraphTensor,
                       attn_cached_k: MPSGraphTensor?,
                       attn_cached_v: MPSGraphTensor?,
                       
                       postattn_norm_weight: MPSGraphTensor,
                       
                       mlp_1_weight: MPSGraphTensor,
                       mlp_2_weight: MPSGraphTensor,
                       mlp_3_weight: MPSGraphTensor,
    ) -> (MPSGraphTensor, MPSGraphTensor, MPSGraphTensor) {
        var residual = hidden_states_tensor
        
        var hidden_states = Qwen3_RMSNorm(graph: graph, hidden_states: hidden_states_tensor, weight: inp_norm_weight, name: "Decoder_InpNorm")
        
        let attn_out = Qwen3_Attention(graph: graph,
                                       hidden_states_tensor: hidden_states,
                                       pos_cos: pos_cos,
                                       pos_sin: pos_sin,
                                       q_proj_weight: attn_qproj_weight,
                                       k_proj_weight: attn_kproj_weight,
                                       v_proj_weight: attn_vproj_weight,
                                       o_proj_weight: attn_oproj_weight,
                                       q_norm_weight: attn_qnorm_weight,
                                       k_norm_weight: attn_knorm_weight,
                                       cached_k: attn_cached_k,
                                       cached_v: attn_cached_v)
        hidden_states = attn_out.attn_output
        let cached_k = attn_out.cached_k
        let cached_v = attn_out.cached_v
        
        hidden_states = graph.addition(residual, hidden_states, name: "Decoder_residual+hidden_states")
        residual = hidden_states
        
        hidden_states = Qwen3_RMSNorm(graph: graph, hidden_states: hidden_states, weight: postattn_norm_weight, name: "Decoder_PostAttnNorm")
        
        hidden_states = Qwen3_MLP(graph: graph, hidden_states, mlp_1_weight, mlp_2_weight, mlp_3_weight, name: "Decoder_MLP")
        //        let a = hidden_states
        hidden_states = graph.addition(residual, hidden_states, name: "Decoder_residual+hidden_states")
        
        //        return (a, cached_k, cached_v)
        return (hidden_states, cached_k, cached_v)
    }
    
    // ======== Attention Methods =======
    
    func apply_rotary_pos_emb(
        graph: MPSGraph,
        q: MPSGraphTensor,
        k: MPSGraphTensor,
        pe_cos: MPSGraphTensor,
        pe_sin: MPSGraphTensor) -> (MPSGraphTensor, MPSGraphTensor) {
            // unsqueeze_dim = 1
            // cos = cos.unsqueeze(unsqueeze_dim)
            let cos = graph.expandDims(pe_cos, axes: [1], name: "rotary_pos_emb:unsqueeze_cos")
            // sin = sin.unsqueeze(unsqueeze_dim)
            let sin = graph.expandDims(pe_sin, axes: [1], name: "rotary_pos_emb:unsqueeze_sin")
            // q_embed = (q * cos) + (rotate_half(q) * sin)
            let q_embed: MPSGraphTensor = graph.addition(
                graph.multiplication(q, cos, name: "rotary_pos_emb:q_x_cos"),
                graph.multiplication(
                    self.rotate_half(graph: graph, x: q, name: "rotary_pos_emb:rotate_half_q"),
                    sin, name: "rotary_pos_emb:q_sin"), name: "rotary_pos_emb:q_embed")
            // k_embed = (k * cos) + (rotate_half(k) * sin)
            let k_embed: MPSGraphTensor = graph.addition(
                graph.multiplication(k, cos, name: "rotary_pos_emb:k_x_cos"),
                graph.multiplication(
                    self.rotate_half(graph: graph, x: k, name: "rotary_pos_emb:rotate_half_k"),
                    sin, name: "rotary_pos_emb:k_sin"), name: "rotary_pos_emb:k_embed")
            return (q_embed, k_embed)
        }
    
    func rotate_half(graph: MPSGraph, x: MPSGraphTensor, name: String) -> MPSGraphTensor {
        let s = x.shape!
        let lastDim = s.count - 1
        let half = Int(truncating: s[lastDim]) / 2
        // x1 = x[..., : x.shape[-1] // 2]
        let x1 = graph.sliceTensor(x, dimension: lastDim, start: 0, length: half, name: "rotate_half:x[:half]")
        // x2 = x[..., x.shape[-1] // 2:]
        let x2 = graph.sliceTensor(x, dimension: lastDim, start: half, length: half, name: "rotate_half:x[half:]")
        let negX2 = graph.negative(with: x2, name: "rotate_half:negx2")
        // return torch.cat((-x2, x1), dim=-1)
        return graph.concatTensors([negX2, x1], dimension: lastDim, name: name)
    }
    
    func repeat_kv(graph: MPSGraph, hidden_states: MPSGraphTensor, n_rep: Int) -> MPSGraphTensor {
        // batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        let batch = hidden_states.shape![0]
        let num_key_value_heads = hidden_states.shape![1]
        let slen = hidden_states.shape![2]
        let head_dim = hidden_states.shape![3]
        if (n_rep == 1) {
            return hidden_states
        }
        // hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        let unsqueezed = graph.expandDims(hidden_states, axes: [2], name: "repeat_kv:expand")
        let broadcasted = graph.broadcast(unsqueezed, shape: [batch, num_key_value_heads, n_rep as NSNumber, slen, head_dim], name: "repeat_kv:broadcast")
        // return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        return graph.reshape(broadcasted, shape: [batch, Int(truncating: num_key_value_heads) * n_rep as NSNumber, slen, head_dim], name: "repeat_kv:reshape")
    }
    
    func tril0(graph: MPSGraph, _ L: NSNumber, _ S: NSNumber) -> MPSGraphTensor {
        /*
         e.g. L = 4, S = 4
         1 0 0 0
         1 1 0 0
         1 1 1 0
         1 1 1 1
         */
        let iL = Int(truncating: L)
        let iS = Int(truncating: S)
        var buffer: [Float32] = []
        for i in 0..<iL {
            for j in 0..<iS {
                buffer.append(Float32(j <= i ? 1.0 : 0.0))
            }
        }
        let data = buffer.withUnsafeBufferPointer { ptr in return Data(buffer: ptr) }
        let temp = graph.constant(data, shape: [L, S], dataType: .float32)
        return temp
    }
    
    func sdpa_attention_forward(
        graph: MPSGraph,
        num_key_value_groups: Int? = nil,
        query: MPSGraphTensor,
        key_arg: MPSGraphTensor,
        value_arg: MPSGraphTensor,
        attention_mask: MPSGraphTensor?,
        dropout: Float32?,
        scaling: Float32?,
        is_causal_arg: Bool?,
    ) -> (MPSGraphTensor, MPSGraphTensor, MPSGraphTensor)  {
        // if hasattr(module, "num_key_value_groups"):
        //    key = repeat_kv(key, module.num_key_value_groups)
        //    value = repeat_kv(value, module.num_key_value_groups)
        var key = key_arg
        var value = value_arg
        if let num_key_value_groups = num_key_value_groups {  // not used by GPT2
            key = repeat_kv(graph: graph, hidden_states: key_arg, n_rep: num_key_value_groups)
            value = repeat_kv(graph: graph, hidden_states: value_arg, n_rep: num_key_value_groups)
        }
        
        // causal_mask = attention_mask  # DEFAULT: NONE
        // if is_causal is None:
        //    is_causal = query.shape[2] > 1 and causal_mask is None
        var is_causal = false
        if (is_causal_arg == nil) {
            is_causal = (Int(truncating: query.shape![2]) > 1) && (attention_mask == nil)
        } else {
            is_causal = is_causal_arg!
        }
        
        var (attn_output, al, vl) = scaled_dot_product_attention(
            graph: graph,
            query: query,
            key: key,
            value: value,
            attn_mask: attention_mask,
            dropout_p: dropout,
            scale: scaling,
            is_causal: is_causal)
        // Not using this built-in method bc of weird Dropout impl & no `is_causal` support
        // var  attn_output = graph.scaledDotProductAttention(query: query,
        //                                                 key: key,
        //                                                 value: value,
        //                                                 scale: scaling!,
        //                                                 name: "spda-attn")
        
        // attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = graph.transposeTensor(attn_output, dimension: 1, withDimension: 2, name: "sdpa:attn_output-transpose")
        // return (attn_output)
        return (attn_output, al, vl)
    }
    
    func scaled_dot_product_attention(
        graph: MPSGraph,
        query: MPSGraphTensor,
        key: MPSGraphTensor,
        value: MPSGraphTensor,
        attn_mask: MPSGraphTensor?,
        dropout_p: Float32?,
        scale: Float32?,
        is_causal: Bool,
    ) -> (MPSGraphTensor, MPSGraphTensor, MPSGraphTensor) {
        // L, S = query.size(-2), key.size(-2)
        let L: NSNumber = query.shape![query.shape!.count - 2]
        let S: NSNumber = key.shape![key.shape!.count - 2]
        // scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        let scale_factor: Float32 = (scale == nil ? 1.0 / Float32(sqrt(Float32(truncating: query.shape![query.shape!.count - 1]))) : scale!)
        // attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        var attn_bias = graph.constant(0.0, shape: [L, S], dataType: query.dataType)
        // if is_causal:
        if is_causal {
            // assert attn_mask is None
            assert (attn_mask == nil)
            // temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            // attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            let bool_mask = self.tril0(graph: graph, L, S)
            let buffer = [Float32](repeating: -1e38, count: Int(truncating: L) * Int(truncating: S))  // we want -inf, but good enough for now
            let data = buffer.withUnsafeBufferPointer { ptr in return Data(buffer: ptr) }
            let neg_tensor = graph.constant(data, shape: [L, S], dataType: .float32)
            attn_bias = graph.select(predicate: bool_mask, trueTensor: attn_bias, falseTensor: neg_tensor, name: "sdpa:select-attn_bias")
        }
        
        // attn_weight = query @ key.transpose(-2, -1) * scale_factor
        let keyT = graph.transposeTensor(key,
                                         dimension: key.shape!.count - 2,
                                         withDimension: key.shape!.count - 1,
                                         name: "sdpa:transpose-key")
        
        var attn_weight = graph.matrixMultiplication(primary: query, secondary: keyT, name: "sdpa:attn_scores")
        attn_weight = graph.multiplication(attn_weight, graph.constant(Double(scale_factor), dataType: query.dataType), name: "sdpa:scaled_attn")
        // attn_weight += attn_bias
        attn_weight = graph.addition(attn_weight, attn_bias, name: "sdpa:attn_weight-plus-attn_bias")
        // attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = graph.softMax(with: attn_weight, axis: attn_weight.shape!.count - 1, name: "sdpa:attn_weight-softmax")
        
        // attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        //===attn_weight = graph.dropout(attn_weight, rate: Double(dropout_p == nil ? 0.0 : dropout_p!), name: "attn_weight-dropout")
        // dropout produces nans, can skip temporarily for inference (since dropout = 0.0), but need to resolve for training
        // TODO: use our own implemented dropout
        
        let al = attn_weight
        let vl = value
        // return attn_weight @ value
        //        print()
        //        print()
        let a = al
        let v = vl
        attn_weight = graph.matrixMultiplication(primary: a, secondary: v, name: "attn_output-weight-x-value")  // something here might be wrong?
//        attn_weight = self.MatMul4D_ByHand(graph, a, v)
        
        return (attn_weight, al, vl) // (... al, vl) = temp, debugging tensor vals
    }
    
    // ======= GPT2 Methods =======
    
    func GPT2_MLP(graph: MPSGraph,
                  hidden_states: MPSGraphTensor,
                  c_fc_weight: MPSGraphTensor,
                  c_fc_bias: MPSGraphTensor,
                  c_proj_weight: MPSGraphTensor,
                  c_proj_bias: MPSGraphTensor
    ) -> MPSGraphTensor {
        var h = self.Linear(graph: graph, input: hidden_states, weight: c_fc_weight, bias: c_fc_bias, name: "GPT2_MLP:h_fc")
        h = self.GELU(graph: graph, input: h)
        h = self.Linear(graph: graph, input: h, weight: c_proj_weight, bias: c_proj_bias, name: "GPT2_MLP:h_proj")
        h = self.Dropout(graph: graph, input: h)  // GPT2 dropout val is set to 0.1
        return h
    }
    
    func GPT2_Attention(graph: MPSGraph,
                        hidden_states: MPSGraphTensor,
                        c_attn_weight: MPSGraphTensor,
                        c_attn_bias: MPSGraphTensor,
                        c_proj_weight: MPSGraphTensor,
                        c_proj_bias: MPSGraphTensor,
                        cached_k: MPSGraphTensor?,
                        cached_v: MPSGraphTensor?) -> (attn_output: MPSGraphTensor, cached_k: MPSGraphTensor, cached_v: MPSGraphTensor) {
        // query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        let c_attn = self.Linear(graph: graph, input: hidden_states, weight: c_attn_weight, bias: c_attn_bias, name: "GPT2_attn:c_attn")
        let q_k_v = graph.split(c_attn, numSplits: 3, axis: 2, name: "GPT2_attn:q_k_v-split")
        var (query_states, key_states, value_states) = (q_k_v[0], q_k_v[1], q_k_v[2])
        
        let head_dim: NSNumber = 64  // TEMP: from default config
        // shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        let shape_q = query_states.shape!.dropLast() + [-1, head_dim] as [NSNumber]
        // shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        let shape_kv = key_states.shape!.dropLast() + [-1, head_dim] as [NSNumber]
        
        // query_states = query_states.view(shape_q).transpose(1, 2)
        query_states = graph.transposeTensor(graph.reshape(query_states, shape: shape_q, name: "GPT2_attn:q-view"),
                                             dimension: 1, withDimension: 2, name: "GPT2_attn:q-transpose")
        // key_states = key_states.view(shape_kv).transpose(1, 2)
        key_states = graph.transposeTensor(graph.reshape(key_states, shape: shape_kv, name: "GPT2_attnn:k-view"),
                                           dimension: 1, withDimension: 2, name: "GPT2_attn:k-transpose")
        // value_states = value_states.view(shape_kv).transpose(1, 2)
        value_states = graph.transposeTensor(graph.reshape(value_states, shape: shape_kv, name: "GPT2_attn:v-view"),
                                             dimension: 1, withDimension: 2, name: "GPT2_attn:v-transpose")
        
        // if layer_past is not None:
        if (cached_k != nil && cached_v != nil) {
            // past_key, past_value = layer_past
            // key_states = torch.cat((past_key, key_states), dim=-2)
            // value_states = torch.cat((past_value, value_states), dim=-2)
            key_states = graph.concatTensors([cached_k!, key_states], dimension: cached_k!.shape!.count - 2, name: "GPT2_attn:add-cached_k")
            value_states = graph.concatTensors([cached_v!, value_states], dimension: cached_v!.shape!.count - 2, name: "GPT2_attn:add-cached_v")
        }
        
        // is_causal = (1 < query_states.shape[-2])
        let is_causal: Bool = 1 < Int(truncating: query_states.shape![query_states.shape!.count - 2])
        
        // attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]  # spda by default :D
        // attn_output, attn_weights = attention_interface(  # is sdpa by default :D (no need to re-implement much)
        //     self,
        //     query_states,
        //     key_states,
        //     value_states,
        //     None, # attention_mask,  # (None by default)
        //     # head_mask=head_mask,  # (None by default)
        //     dropout=self.attn_dropout.p if self.training else 0.0,
        //     is_causal=is_causal,
        //     **kwargs,
        // )
        var (attn_output, _, _) = sdpa_attention_forward(graph: graph,
                                                         query: query_states,
                                                         key_arg: key_states,
                                                         value_arg: value_states,
                                                         attention_mask: nil,
                                                         dropout: 0.0,  // TODO: sdpa dropout
                                                         scaling: 1.0 / sqrt(Float32(truncating: head_dim)),
                                                         is_causal_arg: is_causal)
        
        // attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = graph.reshape(attn_output, shape: attn_output.shape!.dropLast().dropLast() + [-1] as [NSNumber], name: "GPT2_attn:output-reshape")
        
        // attn_output = self.c_proj(attn_ouput)
        attn_output = self.Linear(graph: graph, input: attn_output, weight: c_proj_weight, bias: c_proj_bias, name: "GPT2_attn:c_proj")
        
        // attn_output: torch.Tensor = self.resid_dropout(attn_output)
        // TODO: impl
        
        // present = (key_states, value_states)
        // outputs = (attn_output, present)
        return (attn_output: attn_output, cached_k: key_states, cached_v: value_states)
    }
    
    func GPT2_Block(graph: MPSGraph,
                    hidden_states: MPSGraphTensor,
                    
                    ln1_weight: MPSGraphTensor,
                    ln1_bias: MPSGraphTensor,
                    
                    attn_c_attn_weight: MPSGraphTensor,
                    attn_c_attn_bias: MPSGraphTensor,
                    attn_c_proj_weight: MPSGraphTensor,
                    attn_c_proj_bias: MPSGraphTensor,
                    attn_cached_k: MPSGraphTensor? = nil,
                    attn_cached_v: MPSGraphTensor? = nil,
                    
                    ln2_weight: MPSGraphTensor,
                    ln2_bias: MPSGraphTensor,

                    mlp_c_fc_weight: MPSGraphTensor,
                    mlp_c_fc_bias: MPSGraphTensor,
                    mlp_c_proj_weight: MPSGraphTensor,
                    mlp_c_proj_bias: MPSGraphTensor
    ) -> (output: MPSGraphTensor, cached_k: MPSGraphTensor, cached_v: MPSGraphTensor) {
        let normalizedShape = [768] as [NSNumber]
        
        var residual = hidden_states
        
        var hs = self.LayerNorm(graph: graph,
                                normalizedShape: normalizedShape, input: hidden_states, weight: ln1_weight, bias: ln1_bias, epsilon: 1e-5,
                                name: "GPT2_Block:ln1")
        
        let attn_outputs = self.GPT2_Attention(graph: graph,
                                               hidden_states: hs,
                                               c_attn_weight: attn_c_attn_weight,
                                               c_attn_bias: attn_c_attn_bias,
                                               c_proj_weight: attn_c_proj_weight,
                                               c_proj_bias: attn_c_proj_bias,
                                               cached_k: attn_cached_k,
                                               cached_v: attn_cached_v)
        let attn_output = attn_outputs.attn_output
        let (cached_k, cached_v) = (attn_outputs.cached_k, attn_outputs.cached_v)
        
        hs = graph.addition(attn_output, residual, name: "GPT2_Block:residual-add1")
        
        residual = hs
        hs = self.LayerNorm(graph: graph, normalizedShape: normalizedShape, input: hs, weight: ln2_weight, bias: ln2_bias, epsilon: 1e-5,
                            name: "GPT2_Block:ln2")
        let feed_forward_hidden_states = self.GPT2_MLP(graph: graph,
                                                       hidden_states: hs,
                                                       c_fc_weight: mlp_c_fc_weight,
                                                       c_fc_bias: mlp_c_fc_bias,
                                                       c_proj_weight: mlp_c_proj_weight,
                                                       c_proj_bias: mlp_c_proj_bias)
        hs = graph.addition(residual, feed_forward_hidden_states, name: "GPT2_Block:residual-add2")

        return (output: hs, cached_k: cached_k, cached_v: cached_v)
    }
    
    // ======= ResNet Methods =======
    
    func ResNet_BasicBlock(graph: MPSGraph,
                           input: MPSGraphTensor,
                           c1weight: MPSGraphTensor, bn1weight: MPSGraphTensor, bn1bias: MPSGraphTensor, bn1rm: MPSGraphTensor, bn1rv: MPSGraphTensor,
                           c2weight: MPSGraphTensor, bn2weight: MPSGraphTensor, bn2bias: MPSGraphTensor, bn2rm: MPSGraphTensor, bn2rv: MPSGraphTensor,
                           name: String, frozenRunningParams: Bool = true
    ) -> MPSGraphTensor {
        // BasicBlock(
        //   (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        //   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        //   (relu): ReLU(inplace=True)
        //   (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        //   (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        // )
        let c1 = self.Conv2D(graph: graph, input: input, weight: c1weight, name: name + ":conv1")
        let bn1 = self.BatchNorm2D(graph: graph, input: c1, weight: bn1weight, bias: bn1bias, name: name + ":bn1",
                                   frozenRunningParams: frozenRunningParams, runningMean: bn1rm, runningVar: bn1rv)
        let r1 = self.ReLU(graph: graph, input: bn1, name: name + ":relu1")
        let c2 = self.Conv2D(graph: graph, input: r1, weight: c2weight, name: name + ":conv2")
        let bn2 = self.BatchNorm2D(graph: graph, input: c2, weight: bn2weight, bias: bn2bias, name: name + ":bn2",
                                   frozenRunningParams: frozenRunningParams, runningMean: bn2rm, runningVar: bn2rv)
        let i = graph.addition(bn2, input, name: name + ":addIdentity")
        let r2 = self.ReLU(graph: graph, input: i, name: name + ":relu2")
        return r2
    }
    
    func ResNet_BasicBlockWithDownsample(graph: MPSGraph,
                                         input: MPSGraphTensor,
                                         c1weight: MPSGraphTensor, bn1weight: MPSGraphTensor, bn1bias: MPSGraphTensor, bn1rm: MPSGraphTensor, bn1rv: MPSGraphTensor,
                                         c2weight: MPSGraphTensor, bn2weight: MPSGraphTensor, bn2bias: MPSGraphTensor, bn2rm: MPSGraphTensor, bn2rv: MPSGraphTensor,
                                         ds_cw: MPSGraphTensor, ds_bnw: MPSGraphTensor, ds_bnb: MPSGraphTensor, ds_bnrm: MPSGraphTensor, ds_bnrv: MPSGraphTensor,
                                         name: String, frozenRunningParams: Bool = true
    ) -> MPSGraphTensor {
        // temp; all ResNet w/ downsample use 2x2 stride by default
        let c1 = self.Conv2D(graph: graph, input: input, weight: c1weight, name: name + ":conv1", strideX: 2, strideY: 2)
        let bn1 = self.BatchNorm2D(graph: graph, input: c1, weight: bn1weight, bias: bn1bias, name: name + ":bn1",
                                   frozenRunningParams: frozenRunningParams, runningMean: bn1rm, runningVar: bn1rv)
        let r1 = self.ReLU(graph: graph, input: bn1, name: name + ":relu1")
        let c2 = self.Conv2D(graph: graph, input: r1, weight: c2weight, name: name + ":conv2")
        let bn2 = self.BatchNorm2D(graph: graph, input: c2, weight: bn2weight, bias: bn2bias, name: name + ":bn2",
                                   frozenRunningParams: frozenRunningParams, runningMean: bn2rm, runningVar: bn2rv)
        
        // Downsample
        let dc = self.Conv2D(graph: graph, input: input, weight: ds_cw, name: name + ":downsample_conv", strideX: 2, strideY: 2, paddingX: 0, paddingY: 0)
        let dbn = self.BatchNorm2D(graph: graph, input: dc, weight: ds_bnw, bias: ds_bnb, name: name + ":downsample_bn",
                                   frozenRunningParams: frozenRunningParams, runningMean: ds_bnrm, runningVar: ds_bnrv)
        
        let i = graph.addition(bn2, dbn, name: name + ":addIdentity")
        let r2 = self.ReLU(graph: graph, input: i, name: name + ":relu2")
        return r2
    }
    
    func ResNet_Model(graph: MPSGraph,
                      input: MPSGraphTensor,
                      
                      c1w: MPSGraphTensor, bn1w: MPSGraphTensor, bn1b: MPSGraphTensor, bn1rm: MPSGraphTensor, bn1rv: MPSGraphTensor,
                      
                      l1_b0_c1w: MPSGraphTensor, l1_b0_bn1w: MPSGraphTensor, l1_b0_bn1b: MPSGraphTensor, l1_b0_bn1rm: MPSGraphTensor, l1_b0_bn1rv: MPSGraphTensor,
                      l1_b0_c2w: MPSGraphTensor, l1_b0_bn2w: MPSGraphTensor, l1_b0_bn2b: MPSGraphTensor, l1_b0_bn2rm: MPSGraphTensor, l1_b0_bn2rv: MPSGraphTensor,
                      l1_b1_c1w: MPSGraphTensor, l1_b1_bn1w: MPSGraphTensor, l1_b1_bn1b: MPSGraphTensor, l1_b1_bn1rm: MPSGraphTensor, l1_b1_bn1rv: MPSGraphTensor,
                      l1_b1_c2w: MPSGraphTensor, l1_b1_bn2w: MPSGraphTensor, l1_b1_bn2b: MPSGraphTensor, l1_b1_bn2rm: MPSGraphTensor, l1_b1_bn2rv: MPSGraphTensor,
                      l1_b2_c1w: MPSGraphTensor, l1_b2_bn1w: MPSGraphTensor, l1_b2_bn1b: MPSGraphTensor, l1_b2_bn1rm: MPSGraphTensor, l1_b2_bn1rv: MPSGraphTensor,
                      l1_b2_c2w: MPSGraphTensor, l1_b2_bn2w: MPSGraphTensor, l1_b2_bn2b: MPSGraphTensor, l1_b2_bn2rm: MPSGraphTensor, l1_b2_bn2rv: MPSGraphTensor,
                      
                      l2_b0_c1w: MPSGraphTensor, l2_b0_bn1w: MPSGraphTensor, l2_b0_bn1b: MPSGraphTensor, l2_b0_bn1rm: MPSGraphTensor, l2_b0_bn1rv: MPSGraphTensor,
                      l2_b0_c2w: MPSGraphTensor, l2_b0_bn2w: MPSGraphTensor, l2_b0_bn2b: MPSGraphTensor, l2_b0_bn2rm: MPSGraphTensor, l2_b0_bn2rv: MPSGraphTensor,
                      l2_b0_ds_cw: MPSGraphTensor, l2_b0_ds_bnw: MPSGraphTensor, l2_b0_ds_bnb: MPSGraphTensor, l2_b0_ds_bnrm: MPSGraphTensor, l2_b0_ds_bnrv: MPSGraphTensor,
                      l2_b1_c1w: MPSGraphTensor, l2_b1_bn1w: MPSGraphTensor, l2_b1_bn1b: MPSGraphTensor, l2_b1_bn1rm: MPSGraphTensor, l2_b1_bn1rv: MPSGraphTensor,
                      l2_b1_c2w: MPSGraphTensor, l2_b1_bn2w: MPSGraphTensor, l2_b1_bn2b: MPSGraphTensor, l2_b1_bn2rm: MPSGraphTensor, l2_b1_bn2rv: MPSGraphTensor,
                      l2_b2_c1w: MPSGraphTensor, l2_b2_bn1w: MPSGraphTensor, l2_b2_bn1b: MPSGraphTensor, l2_b2_bn1rm: MPSGraphTensor, l2_b2_bn1rv: MPSGraphTensor,
                      l2_b2_c2w: MPSGraphTensor, l2_b2_bn2w: MPSGraphTensor, l2_b2_bn2b: MPSGraphTensor, l2_b2_bn2rm: MPSGraphTensor, l2_b2_bn2rv: MPSGraphTensor,
                      l2_b3_c1w: MPSGraphTensor, l2_b3_bn1w: MPSGraphTensor, l2_b3_bn1b: MPSGraphTensor, l2_b3_bn1rm: MPSGraphTensor, l2_b3_bn1rv: MPSGraphTensor,
                      l2_b3_c2w: MPSGraphTensor, l2_b3_bn2w: MPSGraphTensor, l2_b3_bn2b: MPSGraphTensor, l2_b3_bn2rm: MPSGraphTensor, l2_b3_bn2rv: MPSGraphTensor,
                      
                      l3_b0_c1w: MPSGraphTensor, l3_b0_bn1w: MPSGraphTensor, l3_b0_bn1b: MPSGraphTensor, l3_b0_bn1rm: MPSGraphTensor, l3_b0_bn1rv: MPSGraphTensor,
                      l3_b0_c2w: MPSGraphTensor, l3_b0_bn2w: MPSGraphTensor, l3_b0_bn2b: MPSGraphTensor, l3_b0_bn2rm: MPSGraphTensor, l3_b0_bn2rv: MPSGraphTensor,
                      l3_b0_ds_cw: MPSGraphTensor, l3_b0_ds_bnw: MPSGraphTensor, l3_b0_ds_bnb: MPSGraphTensor, l3_b0_ds_bnrm: MPSGraphTensor, l3_b0_ds_bnrv: MPSGraphTensor,
                      l3_b1_c1w: MPSGraphTensor, l3_b1_bn1w: MPSGraphTensor, l3_b1_bn1b: MPSGraphTensor, l3_b1_bn1rm: MPSGraphTensor, l3_b1_bn1rv: MPSGraphTensor,
                      l3_b1_c2w: MPSGraphTensor, l3_b1_bn2w: MPSGraphTensor, l3_b1_bn2b: MPSGraphTensor, l3_b1_bn2rm: MPSGraphTensor, l3_b1_bn2rv: MPSGraphTensor,
                      l3_b2_c1w: MPSGraphTensor, l3_b2_bn1w: MPSGraphTensor, l3_b2_bn1b: MPSGraphTensor, l3_b2_bn1rm: MPSGraphTensor, l3_b2_bn1rv: MPSGraphTensor,
                      l3_b2_c2w: MPSGraphTensor, l3_b2_bn2w: MPSGraphTensor, l3_b2_bn2b: MPSGraphTensor, l3_b2_bn2rm: MPSGraphTensor, l3_b2_bn2rv: MPSGraphTensor,
                      l3_b3_c1w: MPSGraphTensor, l3_b3_bn1w: MPSGraphTensor, l3_b3_bn1b: MPSGraphTensor, l3_b3_bn1rm: MPSGraphTensor, l3_b3_bn1rv: MPSGraphTensor,
                      l3_b3_c2w: MPSGraphTensor, l3_b3_bn2w: MPSGraphTensor, l3_b3_bn2b: MPSGraphTensor, l3_b3_bn2rm: MPSGraphTensor, l3_b3_bn2rv: MPSGraphTensor,
                      l3_b4_c1w: MPSGraphTensor, l3_b4_bn1w: MPSGraphTensor, l3_b4_bn1b: MPSGraphTensor, l3_b4_bn1rm: MPSGraphTensor, l3_b4_bn1rv: MPSGraphTensor,
                      l3_b4_c2w: MPSGraphTensor, l3_b4_bn2w: MPSGraphTensor, l3_b4_bn2b: MPSGraphTensor, l3_b4_bn2rm: MPSGraphTensor, l3_b4_bn2rv: MPSGraphTensor,
                      l3_b5_c1w: MPSGraphTensor, l3_b5_bn1w: MPSGraphTensor, l3_b5_bn1b: MPSGraphTensor, l3_b5_bn1rm: MPSGraphTensor, l3_b5_bn1rv: MPSGraphTensor,
                      l3_b5_c2w: MPSGraphTensor, l3_b5_bn2w: MPSGraphTensor, l3_b5_bn2b: MPSGraphTensor, l3_b5_bn2rm: MPSGraphTensor, l3_b5_bn2rv: MPSGraphTensor,

                      l4_b0_c1w: MPSGraphTensor, l4_b0_bn1w: MPSGraphTensor, l4_b0_bn1b: MPSGraphTensor, l4_b0_bn1rm: MPSGraphTensor, l4_b0_bn1rv: MPSGraphTensor,
                      l4_b0_c2w: MPSGraphTensor, l4_b0_bn2w: MPSGraphTensor, l4_b0_bn2b: MPSGraphTensor, l4_b0_bn2rm: MPSGraphTensor, l4_b0_bn2rv: MPSGraphTensor,
                      l4_b0_ds_cw: MPSGraphTensor, l4_b0_ds_bnw: MPSGraphTensor, l4_b0_ds_bnb: MPSGraphTensor, l4_b0_ds_bnrm: MPSGraphTensor, l4_b0_ds_bnrv: MPSGraphTensor,
                      l4_b1_c1w: MPSGraphTensor, l4_b1_bn1w: MPSGraphTensor, l4_b1_bn1b: MPSGraphTensor, l4_b1_bn1rm: MPSGraphTensor, l4_b1_bn1rv: MPSGraphTensor,
                      l4_b1_c2w: MPSGraphTensor, l4_b1_bn2w: MPSGraphTensor, l4_b1_bn2b: MPSGraphTensor, l4_b1_bn2rm: MPSGraphTensor, l4_b1_bn2rv: MPSGraphTensor,
                      l4_b2_c1w: MPSGraphTensor, l4_b2_bn1w: MPSGraphTensor, l4_b2_bn1b: MPSGraphTensor, l4_b2_bn1rm: MPSGraphTensor, l4_b2_bn1rv: MPSGraphTensor,
                      l4_b2_c2w: MPSGraphTensor, l4_b2_bn2w: MPSGraphTensor, l4_b2_bn2b: MPSGraphTensor, l4_b2_bn2rm: MPSGraphTensor, l4_b2_bn2rv: MPSGraphTensor,
                      
                      fcw: MPSGraphTensor, fcb: MPSGraphTensor,
                      
                      training: Bool = false
    ) -> MPSGraphTensor {
        /*
         Modeled off of `torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)`
         
         ResNet(
           (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
           (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
           (relu): ReLU(inplace=True)
         
           (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
         
           (layer1): Sequential(
             (0): BasicBlock(
               (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             )
             (1): BasicBlock(
               (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             )
             (2): BasicBlock(
               (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             )
           )
         
           (layer2): Sequential(
             (0): BasicBlock(
               (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (downsample): Sequential(
                 (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                 (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               )
             )
             (1): BasicBlock(
               (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             )
             (2): BasicBlock(
               (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             )
             (3): BasicBlock(
               (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             )
           )
         
           (layer3): Sequential(
             (0): BasicBlock(
               (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (downsample): Sequential(
                 (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                 (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               )
             )
             (1): BasicBlock(
               (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             )
             (2): BasicBlock(
               (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             )
             (3): BasicBlock(
               (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             )
             (4): BasicBlock(
               (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             )
             (5): BasicBlock(
               (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             )
           )
         
           (layer4): Sequential(
             (0): BasicBlock(
               (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (downsample): Sequential(
                 (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                 (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               )
             )
             (1): BasicBlock(
               (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             )
             (2): BasicBlock(
               (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
               (relu): ReLU(inplace=True)
               (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
               (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             )
           )
         
           (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
         
           (fc): Linear(in_features=512, out_features=1000, bias=True)
         )
         */
        
        let c1 = self.Conv2D(graph: graph, input: input, weight: c1w, name: "ResNet.conv1", strideX: 2, strideY: 2, paddingX: 3, paddingY: 3)
        let bn1 = self.BatchNorm2D(graph: graph, input: c1, weight: bn1w, bias: bn1b, name: "ResNet.bn1", frozenRunningParams: training, runningMean: bn1rm, runningVar: bn1rv)
        let r1 = self.ReLU(graph: graph, input: bn1, name: "ResNet.relu")
        let mp = self.MaxPool2D(graph: graph, input: r1, name: "ResNet.maxpool")
        
        // === Layer 1 ===
        let l1_b0 = self.ResNet_BasicBlock(graph: graph, input: mp, c1weight: l1_b0_c1w, bn1weight: l1_b0_bn1w, bn1bias: l1_b0_bn1b,
                                           bn1rm: l1_b0_bn1rm, bn1rv: l1_b0_bn1rv,
                                           c2weight: l1_b0_c2w, bn2weight: l1_b0_bn2w, bn2bias: l1_b0_bn2b,
                                           bn2rm: l1_b0_bn2rm, bn2rv: l1_b0_bn2rv,
                                           name: "ResNet.layer1[0]", frozenRunningParams: training)
        let l1_b1 = self.ResNet_BasicBlock(graph: graph, input: l1_b0, c1weight: l1_b1_c1w, bn1weight: l1_b1_bn1w, bn1bias: l1_b1_bn1b,
                                           bn1rm: l1_b1_bn1rm, bn1rv: l1_b1_bn1rv,
                                           c2weight: l1_b1_c2w, bn2weight: l1_b1_bn2w, bn2bias: l1_b1_bn2b,
                                           bn2rm: l1_b1_bn2rm, bn2rv: l1_b1_bn2rv,
                                           name: "ResNet.layer1[1]", frozenRunningParams: training)
        let l1_b2 = self.ResNet_BasicBlock(graph: graph, input: l1_b1, c1weight: l1_b2_c1w, bn1weight: l1_b2_bn1w, bn1bias: l1_b2_bn1b,
                                           bn1rm: l1_b2_bn1rm, bn1rv: l1_b2_bn1rv,
                                           c2weight: l1_b2_c2w, bn2weight: l1_b2_bn2w, bn2bias: l1_b2_bn2b,
                                           bn2rm: l1_b2_bn2rm, bn2rv: l1_b2_bn2rv,
                                           name: "ResNet.layer1[2]", frozenRunningParams: training)
        
        // === Layer 2 ===
        let l2_b0 = self.ResNet_BasicBlockWithDownsample(graph: graph, input: l1_b2, c1weight: l2_b0_c1w, bn1weight: l2_b0_bn1w, bn1bias: l2_b0_bn1b,
                                                         bn1rm: l2_b0_bn1rm, bn1rv: l2_b0_bn1rv,
                                                         c2weight: l2_b0_c2w, bn2weight: l2_b0_bn2w, bn2bias: l2_b0_bn2b,
                                                         bn2rm: l2_b0_bn2rm, bn2rv: l2_b0_bn2rv,
                                                         ds_cw: l2_b0_ds_cw, ds_bnw: l2_b0_ds_bnw, ds_bnb: l2_b0_ds_bnb,
                                                         ds_bnrm: l2_b0_ds_bnrm, ds_bnrv: l2_b0_ds_bnrv,
                                                         name: "ResNet.layer2[0]", frozenRunningParams: training)
        
        let l2_b1 = self.ResNet_BasicBlock(graph: graph, input: l2_b0, c1weight: l2_b1_c1w, bn1weight: l2_b1_bn1w, bn1bias: l2_b1_bn1b,
                                           bn1rm: l2_b1_bn1rm, bn1rv: l2_b1_bn1rv,
                                           c2weight: l2_b1_c2w, bn2weight: l2_b1_bn2w, bn2bias: l2_b1_bn2b,
                                           bn2rm: l2_b1_bn2rm, bn2rv: l2_b1_bn2rv,
                                           name: "ResNet.layer2[1]", frozenRunningParams: training)
        let l2_b2 = self.ResNet_BasicBlock(graph: graph, input: l2_b1, c1weight: l2_b2_c1w, bn1weight: l2_b2_bn1w, bn1bias: l2_b2_bn1b,
                                           bn1rm: l2_b2_bn1rm, bn1rv: l2_b2_bn1rv,
                                           c2weight: l2_b2_c2w, bn2weight: l2_b2_bn2w, bn2bias: l2_b2_bn2b,
                                           bn2rm: l2_b2_bn2rm, bn2rv: l2_b2_bn2rv,
                                           name: "ResNet.layer2[2]", frozenRunningParams: training)
        let l2_b3 = self.ResNet_BasicBlock(graph: graph, input: l2_b2, c1weight: l2_b3_c1w, bn1weight: l2_b3_bn1w, bn1bias: l2_b3_bn1b,
                                           bn1rm: l2_b3_bn1rm, bn1rv: l2_b3_bn1rv,
                                           c2weight: l2_b3_c2w, bn2weight: l2_b3_bn2w, bn2bias: l2_b3_bn2b,
                                           bn2rm: l2_b3_bn2rm, bn2rv: l2_b3_bn2rv,
                                           name: "ResNet.layer2[3]", frozenRunningParams: training)
        
        // === Layer 3 ===
        let l3_b0 = self.ResNet_BasicBlockWithDownsample(graph: graph, input: l2_b3, c1weight: l3_b0_c1w, bn1weight: l3_b0_bn1w, bn1bias: l3_b0_bn1b,
                                                         bn1rm: l3_b0_bn1rm, bn1rv: l3_b0_bn1rv,
                                                         c2weight: l3_b0_c2w, bn2weight: l3_b0_bn2w, bn2bias: l3_b0_bn2b,
                                                         bn2rm: l3_b0_bn2rm, bn2rv: l3_b0_bn2rv,
                                                         ds_cw: l3_b0_ds_cw, ds_bnw: l3_b0_ds_bnw, ds_bnb: l3_b0_ds_bnb,
                                                         ds_bnrm: l3_b0_ds_bnrm, ds_bnrv: l3_b0_ds_bnrv,
                                                         name: "ResNet.layer3[0]", frozenRunningParams: training)
        let l3_b1 = self.ResNet_BasicBlock(graph: graph, input: l3_b0, c1weight: l3_b1_c1w, bn1weight: l3_b1_bn1w, bn1bias: l3_b1_bn1b,
                                           bn1rm: l3_b1_bn1rm, bn1rv: l3_b1_bn1rv,
                                           c2weight: l3_b1_c2w, bn2weight: l3_b1_bn2w, bn2bias: l3_b1_bn2b,
                                           bn2rm: l3_b1_bn2rm, bn2rv: l3_b1_bn2rv,
                                           name: "ResNet.layer3[1]", frozenRunningParams: training)
        let l3_b2 = self.ResNet_BasicBlock(graph: graph, input: l3_b1, c1weight: l3_b2_c1w, bn1weight: l3_b2_bn1w, bn1bias: l3_b2_bn1b,
                                           bn1rm: l3_b2_bn1rm, bn1rv: l3_b2_bn1rv,
                                           c2weight: l3_b2_c2w, bn2weight: l3_b2_bn2w, bn2bias: l3_b2_bn2b,
                                           bn2rm: l3_b2_bn2rm, bn2rv: l3_b2_bn2rv,
                                           name: "ResNet.layer3[2]", frozenRunningParams: training)
        let l3_b3 = self.ResNet_BasicBlock(graph: graph, input: l3_b2, c1weight: l3_b3_c1w, bn1weight: l3_b3_bn1w, bn1bias: l3_b3_bn1b,
                                           bn1rm: l3_b3_bn1rm, bn1rv: l3_b3_bn1rv,
                                           c2weight: l3_b3_c2w, bn2weight: l3_b3_bn2w, bn2bias: l3_b3_bn2b,
                                           bn2rm: l3_b3_bn2rm, bn2rv: l3_b3_bn2rv,
                                           name: "ResNet.layer3[3]", frozenRunningParams: training)
        let l3_b4 = self.ResNet_BasicBlock(graph: graph, input: l3_b3, c1weight: l3_b4_c1w, bn1weight: l3_b4_bn1w, bn1bias: l3_b4_bn1b,
                                           bn1rm: l3_b4_bn1rm, bn1rv: l3_b4_bn1rv,
                                           c2weight: l3_b4_c2w, bn2weight: l3_b4_bn2w, bn2bias: l3_b4_bn2b,
                                           bn2rm: l3_b4_bn2rm, bn2rv: l3_b4_bn2rv,
                                           name: "ResNet.layer3[4]", frozenRunningParams: training)
        let l3_b5 = self.ResNet_BasicBlock(graph: graph, input: l3_b4, c1weight: l3_b5_c1w, bn1weight: l3_b5_bn1w, bn1bias: l3_b5_bn1b,
                                           bn1rm: l3_b5_bn1rm, bn1rv: l3_b5_bn1rv,
                                           c2weight: l3_b5_c2w, bn2weight: l3_b5_bn2w, bn2bias: l3_b5_bn2b,
                                           bn2rm: l3_b5_bn2rm, bn2rv: l3_b5_bn2rv,
                                           name: "ResNet.layer3[5]", frozenRunningParams: training)

        // === Layer 4 ===
        let l4_b0 = self.ResNet_BasicBlockWithDownsample(graph: graph, input: l3_b5, c1weight: l4_b0_c1w, bn1weight: l4_b0_bn1w, bn1bias: l4_b0_bn1b,
                                                         bn1rm: l4_b0_bn1rm, bn1rv: l4_b0_bn1rv,
                                                         c2weight: l4_b0_c2w, bn2weight: l4_b0_bn2w, bn2bias: l4_b0_bn2b,
                                                         bn2rm: l4_b0_bn2rm, bn2rv: l4_b0_bn2rv,
                                                         ds_cw: l4_b0_ds_cw, ds_bnw: l4_b0_ds_bnw, ds_bnb: l4_b0_ds_bnb,
                                                         ds_bnrm: l4_b0_ds_bnrm, ds_bnrv: l4_b0_ds_bnrv,
                                                         name: "ResNet.layer4[0]", frozenRunningParams: training)
        let l4_b1 = self.ResNet_BasicBlock(graph: graph, input: l4_b0, c1weight: l4_b1_c1w, bn1weight: l4_b1_bn1w, bn1bias: l4_b1_bn1b,
                                           bn1rm: l4_b1_bn1rm, bn1rv: l4_b1_bn1rv,
                                           c2weight: l4_b1_c2w, bn2weight: l4_b1_bn2w, bn2bias: l4_b1_bn2b,
                                           bn2rm: l4_b1_bn2rm, bn2rv: l4_b1_bn2rv,
                                           name: "ResNet.layer4[1]", frozenRunningParams: training)
        let l4_b2 = self.ResNet_BasicBlock(graph: graph, input: l4_b1, c1weight: l4_b2_c1w, bn1weight: l4_b2_bn1w, bn1bias: l4_b2_bn1b,
                                           bn1rm: l4_b2_bn1rm, bn1rv: l4_b2_bn1rv,
                                           c2weight: l4_b2_c2w, bn2weight: l4_b2_bn2w, bn2bias: l4_b2_bn2b,
                                           bn2rm: l4_b2_bn2rm, bn2rv: l4_b2_bn2rv,
                                           name: "ResNet.layer4[2]", frozenRunningParams: training)
        
        let avgpool = self.AdaptiveAvgPool2D1x1(graph: graph, input: l4_b2, name: "ResNet.avgpool")
        
        let avgpoolReshaped = graph.reshape(avgpool, shape: [avgpool.shape![0], avgpool.shape![1]], name: "ResNet.avgpool:reshapeForFC")
        let fcwTransposed = graph.transposeTensor(fcw, dimension: 0, withDimension: 1, name: "ResNet.fcw:transposed")
        
        let fc = self.Linear(graph: graph, input: avgpoolReshaped, weight: fcwTransposed, bias: fcb, name: "ResNet.fc")
        
        return fc
    }
    
    func ResNet_Model(graph: MPSGraph, input: MPSGraphTensor, params: [MPSGraphTensor], training: Bool = false) -> MPSGraphTensor {
        let C = CounterFrom1()
        C.value = -1
        func nxt() -> MPSGraphTensor {
            return params[C.inc()]
        }
        return self.ResNet_Model(graph: graph, input: input,
                                 c1w: nxt(), bn1w: nxt(), bn1b: nxt(), bn1rm: nxt(), bn1rv: nxt(),
                                 
                                 l1_b0_c1w: nxt(), l1_b0_bn1w: nxt(), l1_b0_bn1b: nxt(), l1_b0_bn1rm: nxt(), l1_b0_bn1rv: nxt(),
                                 l1_b0_c2w: nxt(), l1_b0_bn2w: nxt(), l1_b0_bn2b: nxt(), l1_b0_bn2rm: nxt(), l1_b0_bn2rv: nxt(),
                                 l1_b1_c1w: nxt(), l1_b1_bn1w: nxt(), l1_b1_bn1b: nxt(), l1_b1_bn1rm: nxt(), l1_b1_bn1rv: nxt(),
                                 l1_b1_c2w: nxt(), l1_b1_bn2w: nxt(), l1_b1_bn2b: nxt(), l1_b1_bn2rm: nxt(), l1_b1_bn2rv: nxt(),
                                 l1_b2_c1w: nxt(), l1_b2_bn1w: nxt(), l1_b2_bn1b: nxt(), l1_b2_bn1rm: nxt(), l1_b2_bn1rv: nxt(),
                                 l1_b2_c2w: nxt(), l1_b2_bn2w: nxt(), l1_b2_bn2b: nxt(), l1_b2_bn2rm: nxt(), l1_b2_bn2rv: nxt(),
                                 
                                 l2_b0_c1w: nxt(), l2_b0_bn1w: nxt(), l2_b0_bn1b: nxt(), l2_b0_bn1rm: nxt(), l2_b0_bn1rv: nxt(),
                                 l2_b0_c2w: nxt(), l2_b0_bn2w: nxt(), l2_b0_bn2b: nxt(), l2_b0_bn2rm: nxt(), l2_b0_bn2rv: nxt(),
                                 l2_b0_ds_cw: nxt(), l2_b0_ds_bnw: nxt(), l2_b0_ds_bnb: nxt(), l2_b0_ds_bnrm: nxt(), l2_b0_ds_bnrv: nxt(),
                                 l2_b1_c1w: nxt(), l2_b1_bn1w: nxt(), l2_b1_bn1b: nxt(), l2_b1_bn1rm: nxt(), l2_b1_bn1rv: nxt(),
                                 l2_b1_c2w: nxt(), l2_b1_bn2w: nxt(), l2_b1_bn2b: nxt(), l2_b1_bn2rm: nxt(), l2_b1_bn2rv: nxt(),
                                 l2_b2_c1w: nxt(), l2_b2_bn1w: nxt(), l2_b2_bn1b: nxt(), l2_b2_bn1rm: nxt(), l2_b2_bn1rv: nxt(),
                                 l2_b2_c2w: nxt(), l2_b2_bn2w: nxt(), l2_b2_bn2b: nxt(), l2_b2_bn2rm: nxt(), l2_b2_bn2rv: nxt(),
                                 l2_b3_c1w: nxt(), l2_b3_bn1w: nxt(), l2_b3_bn1b: nxt(), l2_b3_bn1rm: nxt(), l2_b3_bn1rv: nxt(),
                                 l2_b3_c2w: nxt(), l2_b3_bn2w: nxt(), l2_b3_bn2b: nxt(), l2_b3_bn2rm: nxt(), l2_b3_bn2rv: nxt(),
                                 
                                 l3_b0_c1w: nxt(), l3_b0_bn1w: nxt(), l3_b0_bn1b: nxt(), l3_b0_bn1rm: nxt(), l3_b0_bn1rv: nxt(),
                                 l3_b0_c2w: nxt(), l3_b0_bn2w: nxt(), l3_b0_bn2b: nxt(), l3_b0_bn2rm: nxt(), l3_b0_bn2rv: nxt(),
                                 l3_b0_ds_cw: nxt(), l3_b0_ds_bnw: nxt(), l3_b0_ds_bnb: nxt(), l3_b0_ds_bnrm: nxt(), l3_b0_ds_bnrv: nxt(),
                                 l3_b1_c1w: nxt(), l3_b1_bn1w: nxt(), l3_b1_bn1b: nxt(), l3_b1_bn1rm: nxt(), l3_b1_bn1rv: nxt(),
                                 l3_b1_c2w: nxt(), l3_b1_bn2w: nxt(), l3_b1_bn2b: nxt(), l3_b1_bn2rm: nxt(), l3_b1_bn2rv: nxt(),
                                 l3_b2_c1w: nxt(), l3_b2_bn1w: nxt(), l3_b2_bn1b: nxt(), l3_b2_bn1rm: nxt(), l3_b2_bn1rv: nxt(),
                                 l3_b2_c2w: nxt(), l3_b2_bn2w: nxt(), l3_b2_bn2b: nxt(), l3_b2_bn2rm: nxt(), l3_b2_bn2rv: nxt(),
                                 l3_b3_c1w: nxt(), l3_b3_bn1w: nxt(), l3_b3_bn1b: nxt(), l3_b3_bn1rm: nxt(), l3_b3_bn1rv: nxt(),
                                 l3_b3_c2w: nxt(), l3_b3_bn2w: nxt(), l3_b3_bn2b: nxt(), l3_b3_bn2rm: nxt(), l3_b3_bn2rv: nxt(),
                                 l3_b4_c1w: nxt(), l3_b4_bn1w: nxt(), l3_b4_bn1b: nxt(), l3_b4_bn1rm: nxt(), l3_b4_bn1rv: nxt(),
                                 l3_b4_c2w: nxt(), l3_b4_bn2w: nxt(), l3_b4_bn2b: nxt(), l3_b4_bn2rm: nxt(), l3_b4_bn2rv: nxt(),
                                 l3_b5_c1w: nxt(), l3_b5_bn1w: nxt(), l3_b5_bn1b: nxt(), l3_b5_bn1rm: nxt(), l3_b5_bn1rv: nxt(),
                                 l3_b5_c2w: nxt(), l3_b5_bn2w: nxt(), l3_b5_bn2b: nxt(), l3_b5_bn2rm: nxt(), l3_b5_bn2rv: nxt(),
                                 
                                 l4_b0_c1w: nxt(), l4_b0_bn1w: nxt(), l4_b0_bn1b: nxt(), l4_b0_bn1rm: nxt(), l4_b0_bn1rv: nxt(),
                                 l4_b0_c2w: nxt(), l4_b0_bn2w: nxt(), l4_b0_bn2b: nxt(), l4_b0_bn2rm: nxt(), l4_b0_bn2rv: nxt(),
                                 l4_b0_ds_cw: nxt(), l4_b0_ds_bnw: nxt(), l4_b0_ds_bnb: nxt(), l4_b0_ds_bnrm: nxt(), l4_b0_ds_bnrv: nxt(),
                                 l4_b1_c1w: nxt(), l4_b1_bn1w: nxt(), l4_b1_bn1b: nxt(), l4_b1_bn1rm: nxt(), l4_b1_bn1rv: nxt(),
                                 l4_b1_c2w: nxt(), l4_b1_bn2w: nxt(), l4_b1_bn2b: nxt(), l4_b1_bn2rm: nxt(), l4_b1_bn2rv: nxt(),
                                 l4_b2_c1w: nxt(), l4_b2_bn1w: nxt(), l4_b2_bn1b: nxt(), l4_b2_bn1rm: nxt(), l4_b2_bn1rv: nxt(),
                                 l4_b2_c2w: nxt(), l4_b2_bn2w: nxt(), l4_b2_bn2b: nxt(), l4_b2_bn2rm: nxt(), l4_b2_bn2rv: nxt(),
                                 
                                 fcw: nxt(), fcb: nxt(),
                                 
                                 training: training)
    }

    func ResNet_S2_PreLayer2(graph: MPSGraph, input: MPSGraphTensor,
        // === Layer2 ===
        // block 0
        l2_b0_c1w: MPSGraphTensor, l2_b0_bn1w: MPSGraphTensor, l2_b0_bn1b: MPSGraphTensor, l2_b0_bn1rm: MPSGraphTensor, l2_b0_bn1rv: MPSGraphTensor,
        l2_b0_c2w: MPSGraphTensor, l2_b0_bn2w: MPSGraphTensor, l2_b0_bn2b: MPSGraphTensor, l2_b0_bn2rm: MPSGraphTensor, l2_b0_bn2rv: MPSGraphTensor,
        l2_b0_ds_cw: MPSGraphTensor, l2_b0_ds_bnw: MPSGraphTensor, l2_b0_ds_bnb: MPSGraphTensor, l2_b0_ds_bnrm: MPSGraphTensor, l2_b0_ds_bnrv: MPSGraphTensor,
        // block 1
        l2_b1_c1w: MPSGraphTensor, l2_b1_bn1w: MPSGraphTensor, l2_b1_bn1b: MPSGraphTensor, l2_b1_bn1rm: MPSGraphTensor, l2_b1_bn1rv: MPSGraphTensor,
        l2_b1_c2w: MPSGraphTensor, l2_b1_bn2w: MPSGraphTensor, l2_b1_bn2b: MPSGraphTensor, l2_b1_bn2rm: MPSGraphTensor, l2_b1_bn2rv: MPSGraphTensor,
        // block 2
        l2_b2_c1w: MPSGraphTensor, l2_b2_bn1w: MPSGraphTensor, l2_b2_bn1b: MPSGraphTensor, l2_b2_bn1rm: MPSGraphTensor, l2_b2_bn1rv: MPSGraphTensor,
        l2_b2_c2w: MPSGraphTensor, l2_b2_bn2w: MPSGraphTensor, l2_b2_bn2b: MPSGraphTensor, l2_b2_bn2rm: MPSGraphTensor, l2_b2_bn2rv: MPSGraphTensor,
        // block 3
        l2_b3_c1w: MPSGraphTensor, l2_b3_bn1w: MPSGraphTensor, l2_b3_bn1b: MPSGraphTensor, l2_b3_bn1rm: MPSGraphTensor, l2_b3_bn1rv: MPSGraphTensor,
        l2_b3_c2w: MPSGraphTensor, l2_b3_bn2w: MPSGraphTensor, l2_b3_bn2b: MPSGraphTensor, l2_b3_bn2rm: MPSGraphTensor, l2_b3_bn2rv: MPSGraphTensor,

        l3_b0_c1w: MPSGraphTensor, l3_b0_bn1w: MPSGraphTensor, l3_b0_bn1b: MPSGraphTensor, l3_b0_bn1rm: MPSGraphTensor, l3_b0_bn1rv: MPSGraphTensor,
        l3_b0_c2w: MPSGraphTensor, l3_b0_bn2w: MPSGraphTensor, l3_b0_bn2b: MPSGraphTensor, l3_b0_bn2rm: MPSGraphTensor, l3_b0_bn2rv: MPSGraphTensor,
        l3_b0_ds_cw: MPSGraphTensor, l3_b0_ds_bnw: MPSGraphTensor, l3_b0_ds_bnb: MPSGraphTensor, l3_b0_ds_bnrm: MPSGraphTensor, l3_b0_ds_bnrv: MPSGraphTensor,
        l3_b1_c1w: MPSGraphTensor, l3_b1_bn1w: MPSGraphTensor, l3_b1_bn1b: MPSGraphTensor, l3_b1_bn1rm: MPSGraphTensor, l3_b1_bn1rv: MPSGraphTensor,
        l3_b1_c2w: MPSGraphTensor, l3_b1_bn2w: MPSGraphTensor, l3_b1_bn2b: MPSGraphTensor, l3_b1_bn2rm: MPSGraphTensor, l3_b1_bn2rv: MPSGraphTensor,
        l3_b2_c1w: MPSGraphTensor, l3_b2_bn1w: MPSGraphTensor, l3_b2_bn1b: MPSGraphTensor, l3_b2_bn1rm: MPSGraphTensor, l3_b2_bn1rv: MPSGraphTensor,
        l3_b2_c2w: MPSGraphTensor, l3_b2_bn2w: MPSGraphTensor, l3_b2_bn2b: MPSGraphTensor, l3_b2_bn2rm: MPSGraphTensor, l3_b2_bn2rv: MPSGraphTensor,
        l3_b3_c1w: MPSGraphTensor, l3_b3_bn1w: MPSGraphTensor, l3_b3_bn1b: MPSGraphTensor, l3_b3_bn1rm: MPSGraphTensor, l3_b3_bn1rv: MPSGraphTensor,
        l3_b3_c2w: MPSGraphTensor, l3_b3_bn2w: MPSGraphTensor, l3_b3_bn2b: MPSGraphTensor, l3_b3_bn2rm: MPSGraphTensor, l3_b3_bn2rv: MPSGraphTensor,
        l3_b4_c1w: MPSGraphTensor, l3_b4_bn1w: MPSGraphTensor, l3_b4_bn1b: MPSGraphTensor, l3_b4_bn1rm: MPSGraphTensor, l3_b4_bn1rv: MPSGraphTensor,
        l3_b4_c2w: MPSGraphTensor, l3_b4_bn2w: MPSGraphTensor, l3_b4_bn2b: MPSGraphTensor, l3_b4_bn2rm: MPSGraphTensor, l3_b4_bn2rv: MPSGraphTensor,
        l3_b5_c1w: MPSGraphTensor, l3_b5_bn1w: MPSGraphTensor, l3_b5_bn1b: MPSGraphTensor, l3_b5_bn1rm: MPSGraphTensor, l3_b5_bn1rv: MPSGraphTensor,
        l3_b5_c2w: MPSGraphTensor, l3_b5_bn2w: MPSGraphTensor, l3_b5_bn2b: MPSGraphTensor, l3_b5_bn2rm: MPSGraphTensor, l3_b5_bn2rv: MPSGraphTensor,

        l4_b0_c1w: MPSGraphTensor, l4_b0_bn1w: MPSGraphTensor, l4_b0_bn1b: MPSGraphTensor, l4_b0_bn1rm: MPSGraphTensor, l4_b0_bn1rv: MPSGraphTensor,
        l4_b0_c2w: MPSGraphTensor, l4_b0_bn2w: MPSGraphTensor, l4_b0_bn2b: MPSGraphTensor, l4_b0_bn2rm: MPSGraphTensor, l4_b0_bn2rv: MPSGraphTensor,
        l4_b0_ds_cw: MPSGraphTensor, l4_b0_ds_bnw: MPSGraphTensor, l4_b0_ds_bnb: MPSGraphTensor, l4_b0_ds_bnrm: MPSGraphTensor, l4_b0_ds_bnrv: MPSGraphTensor,
        l4_b1_c1w: MPSGraphTensor, l4_b1_bn1w: MPSGraphTensor, l4_b1_bn1b: MPSGraphTensor, l4_b1_bn1rm: MPSGraphTensor, l4_b1_bn1rv: MPSGraphTensor,
        l4_b1_c2w: MPSGraphTensor, l4_b1_bn2w: MPSGraphTensor, l4_b1_bn2b: MPSGraphTensor, l4_b1_bn2rm: MPSGraphTensor, l4_b1_bn2rv: MPSGraphTensor,
        l4_b2_c1w: MPSGraphTensor, l4_b2_bn1w: MPSGraphTensor, l4_b2_bn1b: MPSGraphTensor, l4_b2_bn1rm: MPSGraphTensor, l4_b2_bn1rv: MPSGraphTensor,
        l4_b2_c2w: MPSGraphTensor, l4_b2_bn2w: MPSGraphTensor, l4_b2_bn2b: MPSGraphTensor, l4_b2_bn2rm: MPSGraphTensor, l4_b2_bn2rv: MPSGraphTensor,

        fcw: MPSGraphTensor, fcb: MPSGraphTensor,

        training: Bool
    ) -> MPSGraphTensor {
        // === Layer 2 ===
        let l2_b0 = self.ResNet_BasicBlockWithDownsample(graph: graph, input: input, c1weight: l2_b0_c1w, bn1weight: l2_b0_bn1w, bn1bias: l2_b0_bn1b,
                                                         bn1rm: l2_b0_bn1rm, bn1rv: l2_b0_bn1rv,
                                                         c2weight: l2_b0_c2w, bn2weight: l2_b0_bn2w, bn2bias: l2_b0_bn2b,
                                                         bn2rm: l2_b0_bn2rm, bn2rv: l2_b0_bn2rv,
                                                         ds_cw: l2_b0_ds_cw, ds_bnw: l2_b0_ds_bnw, ds_bnb: l2_b0_ds_bnb,
                                                         ds_bnrm: l2_b0_ds_bnrm, ds_bnrv: l2_b0_ds_bnrv,
                                                         name: "ResNet.layer2[0]", frozenRunningParams: training)
        let l2_b1 = self.ResNet_BasicBlock(graph: graph, input: l2_b0, c1weight: l2_b1_c1w, bn1weight: l2_b1_bn1w, bn1bias: l2_b1_bn1b,
                                           bn1rm: l2_b1_bn1rm, bn1rv: l2_b1_bn1rv,
                                           c2weight: l2_b1_c2w, bn2weight: l2_b1_bn2w, bn2bias: l2_b1_bn2b,
                                           bn2rm: l2_b1_bn2rm, bn2rv: l2_b1_bn2rv,
                                           name: "ResNet.layer2[1]", frozenRunningParams: training)
        let l2_b2 = self.ResNet_BasicBlock(graph: graph, input: l2_b1, c1weight: l2_b2_c1w, bn1weight: l2_b2_bn1w, bn1bias: l2_b2_bn1b,
                                           bn1rm: l2_b2_bn1rm, bn1rv: l2_b2_bn1rv,
                                           c2weight: l2_b2_c2w, bn2weight: l2_b2_bn2w, bn2bias: l2_b2_bn2b,
                                           bn2rm: l2_b2_bn2rm, bn2rv: l2_b2_bn2rv,
                                           name: "ResNet.layer2[2]", frozenRunningParams: training)
        let l2_b3 = self.ResNet_BasicBlock(graph: graph, input: l2_b2, c1weight: l2_b3_c1w, bn1weight: l2_b3_bn1w, bn1bias: l2_b3_bn1b,
                                           bn1rm: l2_b3_bn1rm, bn1rv: l2_b3_bn1rv,
                                           c2weight: l2_b3_c2w, bn2weight: l2_b3_bn2w, bn2bias: l2_b3_bn2b,
                                           bn2rm: l2_b3_bn2rm, bn2rv: l2_b3_bn2rv,
                                           name: "ResNet.layer2[3]", frozenRunningParams: training)
        // === Layer 3 ===
        let l3_b0 = self.ResNet_BasicBlockWithDownsample(graph: graph, input: l2_b3, c1weight: l3_b0_c1w, bn1weight: l3_b0_bn1w, bn1bias: l3_b0_bn1b,
                                                         bn1rm: l3_b0_bn1rm, bn1rv: l3_b0_bn1rv,
                                                         c2weight: l3_b0_c2w, bn2weight: l3_b0_bn2w, bn2bias: l3_b0_bn2b,
                                                         bn2rm: l3_b0_bn2rm, bn2rv: l3_b0_bn2rv,
                                                         ds_cw: l3_b0_ds_cw, ds_bnw: l3_b0_ds_bnw, ds_bnb: l3_b0_ds_bnb,
                                                         ds_bnrm: l3_b0_ds_bnrm, ds_bnrv: l3_b0_ds_bnrv,
                                                         name: "ResNet.layer3[0]", frozenRunningParams: training)
        let l3_b1 = self.ResNet_BasicBlock(graph: graph, input: l3_b0, c1weight: l3_b1_c1w, bn1weight: l3_b1_bn1w, bn1bias: l3_b1_bn1b,
                                           bn1rm: l3_b1_bn1rm, bn1rv: l3_b1_bn1rv,
                                           c2weight: l3_b1_c2w, bn2weight: l3_b1_bn2w, bn2bias: l3_b1_bn2b,
                                           bn2rm: l3_b1_bn2rm, bn2rv: l3_b1_bn2rv,
                                           name: "ResNet.layer3[1]", frozenRunningParams: training)
        let l3_b2 = self.ResNet_BasicBlock(graph: graph, input: l3_b1, c1weight: l3_b2_c1w, bn1weight: l3_b2_bn1w, bn1bias: l3_b2_bn1b,
                                           bn1rm: l3_b2_bn1rm, bn1rv: l3_b2_bn1rv,
                                           c2weight: l3_b2_c2w, bn2weight: l3_b2_bn2w, bn2bias: l3_b2_bn2b,
                                           bn2rm: l3_b2_bn2rm, bn2rv: l3_b2_bn2rv,
                                           name: "ResNet.layer3[2]", frozenRunningParams: training)
        let l3_b3 = self.ResNet_BasicBlock(graph: graph, input: l3_b2, c1weight: l3_b3_c1w, bn1weight: l3_b3_bn1w, bn1bias: l3_b3_bn1b,
                                           bn1rm: l3_b3_bn1rm, bn1rv: l3_b3_bn1rv,
                                           c2weight: l3_b3_c2w, bn2weight: l3_b3_bn2w, bn2bias: l3_b3_bn2b,
                                           bn2rm: l3_b3_bn2rm, bn2rv: l3_b3_bn2rv,
                                           name: "ResNet.layer3[3]", frozenRunningParams: training)
        let l3_b4 = self.ResNet_BasicBlock(graph: graph, input: l3_b3, c1weight: l3_b4_c1w, bn1weight: l3_b4_bn1w, bn1bias: l3_b4_bn1b,
                                           bn1rm: l3_b4_bn1rm, bn1rv: l3_b4_bn1rv,
                                           c2weight: l3_b4_c2w, bn2weight: l3_b4_bn2w, bn2bias: l3_b4_bn2b,
                                           bn2rm: l3_b4_bn2rm, bn2rv: l3_b4_bn2rv,
                                           name: "ResNet.layer3[4]", frozenRunningParams: training)
        let l3_b5 = self.ResNet_BasicBlock(graph: graph, input: l3_b4, c1weight: l3_b5_c1w, bn1weight: l3_b5_bn1w, bn1bias: l3_b5_bn1b,
                                           bn1rm: l3_b5_bn1rm, bn1rv: l3_b5_bn1rv,
                                           c2weight: l3_b5_c2w, bn2weight: l3_b5_bn2w, bn2bias: l3_b5_bn2b,
                                           bn2rm: l3_b5_bn2rm, bn2rv: l3_b5_bn2rv,
                                           name: "ResNet.layer3[5]", frozenRunningParams: training)
        // === Layer 4 ===
        let l4_b0 = self.ResNet_BasicBlockWithDownsample(graph: graph, input: l3_b5, c1weight: l4_b0_c1w, bn1weight: l4_b0_bn1w, bn1bias: l4_b0_bn1b,
                                                         bn1rm: l4_b0_bn1rm, bn1rv: l4_b0_bn1rv,
                                                         c2weight: l4_b0_c2w, bn2weight: l4_b0_bn2w, bn2bias: l4_b0_bn2b,
                                                         bn2rm: l4_b0_bn2rm, bn2rv: l4_b0_bn2rv,
                                                         ds_cw: l4_b0_ds_cw, ds_bnw: l4_b0_ds_bnw, ds_bnb: l4_b0_ds_bnb,
                                                         ds_bnrm: l4_b0_ds_bnrm, ds_bnrv: l4_b0_ds_bnrv,
                                                         name: "ResNet.layer4[0]", frozenRunningParams: training)
        let l4_b1 = self.ResNet_BasicBlock(graph: graph, input: l4_b0, c1weight: l4_b1_c1w, bn1weight: l4_b1_bn1w, bn1bias: l4_b1_bn1b,
                                           bn1rm: l4_b1_bn1rm, bn1rv: l4_b1_bn1rv,
                                           c2weight: l4_b1_c2w, bn2weight: l4_b1_bn2w, bn2bias: l4_b1_bn2b,
                                           bn2rm: l4_b1_bn2rm, bn2rv: l4_b1_bn2rv,
                                           name: "ResNet.layer4[1]", frozenRunningParams: training)
        let l4_b2 = self.ResNet_BasicBlock(graph: graph, input: l4_b1, c1weight: l4_b2_c1w, bn1weight: l4_b2_bn1w, bn1bias: l4_b2_bn1b,
                                           bn1rm: l4_b2_bn1rm, bn1rv: l4_b2_bn1rv,
                                           c2weight: l4_b2_c2w, bn2weight: l4_b2_bn2w, bn2bias: l4_b2_bn2b,
                                           bn2rm: l4_b2_bn2rm, bn2rv: l4_b2_bn2rv,
                                           name: "ResNet.layer4[2]", frozenRunningParams: training)

        let avgpool = self.AdaptiveAvgPool2D1x1(graph: graph, input: l4_b2, name: "ResNet.avgpool")

        let avgpoolReshaped = graph.reshape(avgpool, shape: [avgpool.shape![0], avgpool.shape![1]], name: "ResNet.avgpool:reshapeForFC")
        let fcwTransposed = graph.transposeTensor(fcw, dimension: 0, withDimension: 1, name: "ResNet.fcw:transposed")

        let fc = self.Linear(graph: graph, input: avgpoolReshaped, weight: fcwTransposed, bias: fcb, name: "ResNet.fc")
        return fc
    }

    func ResNet_S2_PreLayer2(graph: MPSGraph, input: MPSGraphTensor, params: [MPSGraphTensor], frozenRunningParams: Bool = true) -> MPSGraphTensor {
        let C = CounterFrom1()
        C.value = -1
        func nxt() -> MPSGraphTensor {
            return params[C.inc()]
        }
        return self.ResNet_S2_PreLayer2(graph: graph, input: input,
                                  l2_b0_c1w: nxt(), l2_b0_bn1w: nxt(), l2_b0_bn1b: nxt(), l2_b0_bn1rm: nxt(), l2_b0_bn1rv: nxt(),
                                  l2_b0_c2w: nxt(), l2_b0_bn2w: nxt(), l2_b0_bn2b: nxt(), l2_b0_bn2rm: nxt(), l2_b0_bn2rv: nxt(),
                                  l2_b0_ds_cw: nxt(), l2_b0_ds_bnw: nxt(), l2_b0_ds_bnb: nxt(), l2_b0_ds_bnrm: nxt(), l2_b0_ds_bnrv: nxt(),
                                  // block 1
                                  l2_b1_c1w: nxt(), l2_b1_bn1w: nxt(), l2_b1_bn1b: nxt(), l2_b1_bn1rm: nxt(), l2_b1_bn1rv: nxt(),
                                  l2_b1_c2w: nxt(), l2_b1_bn2w: nxt(), l2_b1_bn2b: nxt(), l2_b1_bn2rm: nxt(), l2_b1_bn2rv: nxt(),
                                  // block 2
                                  l2_b2_c1w: nxt(), l2_b2_bn1w: nxt(), l2_b2_bn1b: nxt(), l2_b2_bn1rm: nxt(), l2_b2_bn1rv: nxt(),
                                  l2_b2_c2w: nxt(), l2_b2_bn2w: nxt(), l2_b2_bn2b: nxt(), l2_b2_bn2rm: nxt(), l2_b2_bn2rv: nxt(),
                                  // block 3
                                  l2_b3_c1w: nxt(), l2_b3_bn1w: nxt(), l2_b3_bn1b: nxt(), l2_b3_bn1rm: nxt(), l2_b3_bn1rv: nxt(),
                                  l2_b3_c2w: nxt(), l2_b3_bn2w: nxt(), l2_b3_bn2b: nxt(), l2_b3_bn2rm: nxt(), l2_b3_bn2rv: nxt(),

                                 l3_b0_c1w: nxt(), l3_b0_bn1w: nxt(), l3_b0_bn1b: nxt(), l3_b0_bn1rm: nxt(), l3_b0_bn1rv: nxt(),
                                 l3_b0_c2w: nxt(), l3_b0_bn2w: nxt(), l3_b0_bn2b: nxt(), l3_b0_bn2rm: nxt(), l3_b0_bn2rv: nxt(),
                                 l3_b0_ds_cw: nxt(), l3_b0_ds_bnw: nxt(), l3_b0_ds_bnb: nxt(), l3_b0_ds_bnrm: nxt(), l3_b0_ds_bnrv: nxt(),
                                 l3_b1_c1w: nxt(), l3_b1_bn1w: nxt(), l3_b1_bn1b: nxt(), l3_b1_bn1rm: nxt(), l3_b1_bn1rv: nxt(),
                                 l3_b1_c2w: nxt(), l3_b1_bn2w: nxt(), l3_b1_bn2b: nxt(), l3_b1_bn2rm: nxt(), l3_b1_bn2rv: nxt(),
                                 l3_b2_c1w: nxt(), l3_b2_bn1w: nxt(), l3_b2_bn1b: nxt(), l3_b2_bn1rm: nxt(), l3_b2_bn1rv: nxt(),
                                 l3_b2_c2w: nxt(), l3_b2_bn2w: nxt(), l3_b2_bn2b: nxt(), l3_b2_bn2rm: nxt(), l3_b2_bn2rv: nxt(),
                                 l3_b3_c1w: nxt(), l3_b3_bn1w: nxt(), l3_b3_bn1b: nxt(), l3_b3_bn1rm: nxt(), l3_b3_bn1rv: nxt(),
                                 l3_b3_c2w: nxt(), l3_b3_bn2w: nxt(), l3_b3_bn2b: nxt(), l3_b3_bn2rm: nxt(), l3_b3_bn2rv: nxt(),
                                 l3_b4_c1w: nxt(), l3_b4_bn1w: nxt(), l3_b4_bn1b: nxt(), l3_b4_bn1rm: nxt(), l3_b4_bn1rv: nxt(),
                                 l3_b4_c2w: nxt(), l3_b4_bn2w: nxt(), l3_b4_bn2b: nxt(), l3_b4_bn2rm: nxt(), l3_b4_bn2rv: nxt(),
                                 l3_b5_c1w: nxt(), l3_b5_bn1w: nxt(), l3_b5_bn1b: nxt(), l3_b5_bn1rm: nxt(), l3_b5_bn1rv: nxt(),
                                 l3_b5_c2w: nxt(), l3_b5_bn2w: nxt(), l3_b5_bn2b: nxt(), l3_b5_bn2rm: nxt(), l3_b5_bn2rv: nxt(),

                                 l4_b0_c1w: nxt(), l4_b0_bn1w: nxt(), l4_b0_bn1b: nxt(), l4_b0_bn1rm: nxt(), l4_b0_bn1rv: nxt(),
                                 l4_b0_c2w: nxt(), l4_b0_bn2w: nxt(), l4_b0_bn2b: nxt(), l4_b0_bn2rm: nxt(), l4_b0_bn2rv: nxt(),
                                 l4_b0_ds_cw: nxt(), l4_b0_ds_bnw: nxt(), l4_b0_ds_bnb: nxt(), l4_b0_ds_bnrm: nxt(), l4_b0_ds_bnrv: nxt(),
                                 l4_b1_c1w: nxt(), l4_b1_bn1w: nxt(), l4_b1_bn1b: nxt(), l4_b1_bn1rm: nxt(), l4_b1_bn1rv: nxt(),
                                 l4_b1_c2w: nxt(), l4_b1_bn2w: nxt(), l4_b1_bn2b: nxt(), l4_b1_bn2rm: nxt(), l4_b1_bn2rv: nxt(),
                                 l4_b2_c1w: nxt(), l4_b2_bn1w: nxt(), l4_b2_bn1b: nxt(), l4_b2_bn1rm: nxt(), l4_b2_bn1rv: nxt(),
                                 l4_b2_c2w: nxt(), l4_b2_bn2w: nxt(), l4_b2_bn2b: nxt(), l4_b2_bn2rm: nxt(), l4_b2_bn2rv: nxt(),
                                 fcw: nxt(), fcb: nxt(),
                                 training: frozenRunningParams)
    }

    func ResNet_S2_PreLayer3(graph: MPSGraph, input: MPSGraphTensor,
        // === Layer3 ===
        // block 0
        l3_b0_c1w: MPSGraphTensor, l3_b0_bn1w: MPSGraphTensor, l3_b0_bn1b: MPSGraphTensor, l3_b0_bn1rm: MPSGraphTensor, l3_b0_bn1rv: MPSGraphTensor,
        l3_b0_c2w: MPSGraphTensor, l3_b0_bn2w: MPSGraphTensor, l3_b0_bn2b: MPSGraphTensor, l3_b0_bn2rm: MPSGraphTensor, l3_b0_bn2rv: MPSGraphTensor,
        l3_b0_ds_cw: MPSGraphTensor, l3_b0_ds_bnw: MPSGraphTensor, l3_b0_ds_bnb: MPSGraphTensor, l3_b0_ds_bnrm: MPSGraphTensor, l3_b0_ds_bnrv: MPSGraphTensor,
        // block 1
        l3_b1_c1w: MPSGraphTensor, l3_b1_bn1w: MPSGraphTensor, l3_b1_bn1b: MPSGraphTensor, l3_b1_bn1rm: MPSGraphTensor, l3_b1_bn1rv: MPSGraphTensor,
        l3_b1_c2w: MPSGraphTensor, l3_b1_bn2w: MPSGraphTensor, l3_b1_bn2b: MPSGraphTensor, l3_b1_bn2rm: MPSGraphTensor, l3_b1_bn2rv: MPSGraphTensor,
        // block 2
        l3_b2_c1w: MPSGraphTensor, l3_b2_bn1w: MPSGraphTensor, l3_b2_bn1b: MPSGraphTensor, l3_b2_bn1rm: MPSGraphTensor, l3_b2_bn1rv: MPSGraphTensor,
        l3_b2_c2w: MPSGraphTensor, l3_b2_bn2w: MPSGraphTensor, l3_b2_bn2b: MPSGraphTensor, l3_b2_bn2rm: MPSGraphTensor, l3_b2_bn2rv: MPSGraphTensor,
        // block 3
        l3_b3_c1w: MPSGraphTensor, l3_b3_bn1w: MPSGraphTensor, l3_b3_bn1b: MPSGraphTensor, l3_b3_bn1rm: MPSGraphTensor, l3_b3_bn1rv: MPSGraphTensor,
        l3_b3_c2w: MPSGraphTensor, l3_b3_bn2w: MPSGraphTensor, l3_b3_bn2b: MPSGraphTensor, l3_b3_bn2rm: MPSGraphTensor, l3_b3_bn2rv: MPSGraphTensor,
        // block 4
        l3_b4_c1w: MPSGraphTensor, l3_b4_bn1w: MPSGraphTensor, l3_b4_bn1b: MPSGraphTensor, l3_b4_bn1rm: MPSGraphTensor, l3_b4_bn1rv: MPSGraphTensor,
        l3_b4_c2w: MPSGraphTensor, l3_b4_bn2w: MPSGraphTensor, l3_b4_bn2b: MPSGraphTensor, l3_b4_bn2rm: MPSGraphTensor, l3_b4_bn2rv: MPSGraphTensor,
        // block 5
        l3_b5_c1w: MPSGraphTensor, l3_b5_bn1w: MPSGraphTensor, l3_b5_bn1b: MPSGraphTensor, l3_b5_bn1rm: MPSGraphTensor, l3_b5_bn1rv: MPSGraphTensor,
        l3_b5_c2w: MPSGraphTensor, l3_b5_bn2w: MPSGraphTensor, l3_b5_bn2b: MPSGraphTensor, l3_b5_bn2rm: MPSGraphTensor, l3_b5_bn2rv: MPSGraphTensor,

        l4_b0_c1w: MPSGraphTensor, l4_b0_bn1w: MPSGraphTensor, l4_b0_bn1b: MPSGraphTensor, l4_b0_bn1rm: MPSGraphTensor, l4_b0_bn1rv: MPSGraphTensor,
        l4_b0_c2w: MPSGraphTensor, l4_b0_bn2w: MPSGraphTensor, l4_b0_bn2b: MPSGraphTensor, l4_b0_bn2rm: MPSGraphTensor, l4_b0_bn2rv: MPSGraphTensor,
        l4_b0_ds_cw: MPSGraphTensor, l4_b0_ds_bnw: MPSGraphTensor, l4_b0_ds_bnb: MPSGraphTensor, l4_b0_ds_bnrm: MPSGraphTensor, l4_b0_ds_bnrv: MPSGraphTensor,
        l4_b1_c1w: MPSGraphTensor, l4_b1_bn1w: MPSGraphTensor, l4_b1_bn1b: MPSGraphTensor, l4_b1_bn1rm: MPSGraphTensor, l4_b1_bn1rv: MPSGraphTensor,
        l4_b1_c2w: MPSGraphTensor, l4_b1_bn2w: MPSGraphTensor, l4_b1_bn2b: MPSGraphTensor, l4_b1_bn2rm: MPSGraphTensor, l4_b1_bn2rv: MPSGraphTensor,
        l4_b2_c1w: MPSGraphTensor, l4_b2_bn1w: MPSGraphTensor, l4_b2_bn1b: MPSGraphTensor, l4_b2_bn1rm: MPSGraphTensor, l4_b2_bn1rv: MPSGraphTensor,
        l4_b2_c2w: MPSGraphTensor, l4_b2_bn2w: MPSGraphTensor, l4_b2_bn2b: MPSGraphTensor, l4_b2_bn2rm: MPSGraphTensor, l4_b2_bn2rv: MPSGraphTensor,
        
        fcw: MPSGraphTensor, fcb: MPSGraphTensor,
        
        frozenRunningParams: Bool
    ) -> MPSGraphTensor {
        // === Layer 3 ===
        let l3_b0 = self.ResNet_BasicBlockWithDownsample(graph: graph, input: input, c1weight: l3_b0_c1w, bn1weight: l3_b0_bn1w, bn1bias: l3_b0_bn1b,
                                                         bn1rm: l3_b0_bn1rm, bn1rv: l3_b0_bn1rv,
                                                         c2weight: l3_b0_c2w, bn2weight: l3_b0_bn2w, bn2bias: l3_b0_bn2b,
                                                         bn2rm: l3_b0_bn2rm, bn2rv: l3_b0_bn2rv,
                                                         ds_cw: l3_b0_ds_cw, ds_bnw: l3_b0_ds_bnw, ds_bnb: l3_b0_ds_bnb,
                                                         ds_bnrm: l3_b0_ds_bnrm, ds_bnrv: l3_b0_ds_bnrv,
                                                         name: "ResNet.layer3[0]", frozenRunningParams: frozenRunningParams)
        let l3_b1 = self.ResNet_BasicBlock(graph: graph, input: l3_b0, c1weight: l3_b1_c1w, bn1weight: l3_b1_bn1w, bn1bias: l3_b1_bn1b,
                                           bn1rm: l3_b1_bn1rm, bn1rv: l3_b1_bn1rv,
                                           c2weight: l3_b1_c2w, bn2weight: l3_b1_bn2w, bn2bias: l3_b1_bn2b,
                                           bn2rm: l3_b1_bn2rm, bn2rv: l3_b1_bn2rv,
                                           name: "ResNet.layer3[1]", frozenRunningParams: frozenRunningParams)
//        let l3_b1 = input
        let l3_b2 = self.ResNet_BasicBlock(graph: graph, input: l3_b1, c1weight: l3_b2_c1w, bn1weight: l3_b2_bn1w, bn1bias: l3_b2_bn1b,
                                           bn1rm: l3_b2_bn1rm, bn1rv: l3_b2_bn1rv,
                                           c2weight: l3_b2_c2w, bn2weight: l3_b2_bn2w, bn2bias: l3_b2_bn2b,
                                           bn2rm: l3_b2_bn2rm, bn2rv: l3_b2_bn2rv,
                                           name: "ResNet.layer3[1]", frozenRunningParams: frozenRunningParams)
//        let l3_b2 = input
        let l3_b3 = self.ResNet_BasicBlock(graph: graph, input: l3_b2, c1weight: l3_b3_c1w, bn1weight: l3_b3_bn1w, bn1bias: l3_b3_bn1b,
                                           bn1rm: l3_b3_bn1rm, bn1rv: l3_b3_bn1rv,
                                           c2weight: l3_b3_c2w, bn2weight: l3_b3_bn2w, bn2bias: l3_b3_bn2b,
                                           bn2rm: l3_b3_bn2rm, bn2rv: l3_b3_bn2rv,
                                           name: "ResNet.layer3[1]", frozenRunningParams: frozenRunningParams)
//        let l3_b3 = input
        let l3_b4 = self.ResNet_BasicBlock(graph: graph, input: l3_b3, c1weight: l3_b4_c1w, bn1weight: l3_b4_bn1w, bn1bias: l3_b4_bn1b,
                                           bn1rm: l3_b4_bn1rm, bn1rv: l3_b4_bn1rv,
                                           c2weight: l3_b4_c2w, bn2weight: l3_b4_bn2w, bn2bias: l3_b4_bn2b,
                                           bn2rm: l3_b4_bn2rm, bn2rv: l3_b4_bn2rv,
                                           name: "ResNet.layer3[1]", frozenRunningParams: frozenRunningParams)
        let l3_b5 = self.ResNet_BasicBlock(graph: graph, input: l3_b4, c1weight: l3_b5_c1w, bn1weight: l3_b5_bn1w, bn1bias: l3_b5_bn1b,
                                           bn1rm: l3_b5_bn1rm, bn1rv: l3_b5_bn1rv,
                                           c2weight: l3_b5_c2w, bn2weight: l3_b5_bn2w, bn2bias: l3_b5_bn2b,
                                           bn2rm: l3_b5_bn2rm, bn2rv: l3_b5_bn2rv,
                                           name: "ResNet.layer3[1]", frozenRunningParams: frozenRunningParams)
        // === Layer 4 ===
        let l4_b0 = self.ResNet_BasicBlockWithDownsample(graph: graph, input: l3_b5, c1weight: l4_b0_c1w, bn1weight: l4_b0_bn1w, bn1bias: l4_b0_bn1b,
                                                         bn1rm: l4_b0_bn1rm, bn1rv: l4_b0_bn1rv,
                                                         c2weight: l4_b0_c2w, bn2weight: l4_b0_bn2w, bn2bias: l4_b0_bn2b,
                                                         bn2rm: l4_b0_bn2rm, bn2rv: l4_b0_bn2rv,
                                                         ds_cw: l4_b0_ds_cw, ds_bnw: l4_b0_ds_bnw, ds_bnb: l4_b0_ds_bnb,
                                                         ds_bnrm: l4_b0_ds_bnrm, ds_bnrv: l4_b0_ds_bnrv,
                                                         name: "ResNet.layer4[0]", frozenRunningParams: frozenRunningParams)
        let l4_b1 = self.ResNet_BasicBlock(graph: graph, input: l4_b0, c1weight: l4_b1_c1w, bn1weight: l4_b1_bn1w, bn1bias: l4_b1_bn1b,
                                           bn1rm: l4_b1_bn1rm, bn1rv: l4_b1_bn1rv,
                                           c2weight: l4_b1_c2w, bn2weight: l4_b1_bn2w, bn2bias: l4_b1_bn2b,
                                           bn2rm: l4_b1_bn2rm, bn2rv: l4_b1_bn2rv,
                                           name: "ResNet.layer4[1]", frozenRunningParams: frozenRunningParams)
        let l4_b2 = self.ResNet_BasicBlock(graph: graph, input: l4_b1, c1weight: l4_b2_c1w, bn1weight: l4_b2_bn1w, bn1bias: l4_b2_bn1b,
                                           bn1rm: l4_b2_bn1rm, bn1rv: l4_b2_bn1rv,
                                           c2weight: l4_b2_c2w, bn2weight: l4_b2_bn2w, bn2bias: l4_b2_bn2b,
                                           bn2rm: l4_b2_bn2rm, bn2rv: l4_b2_bn2rv,
                                           name: "ResNet.layer4[2]", frozenRunningParams: frozenRunningParams)
        
        let avgpool = self.AdaptiveAvgPool2D1x1(graph: graph, input: l4_b2, name: "ResNet.avgpool")
        
        let avgpoolReshaped = graph.reshape(avgpool, shape: [avgpool.shape![0], avgpool.shape![1]], name: "ResNet.avgpool:reshapeForFC")
        let fcwTransposed = graph.transposeTensor(fcw, dimension: 0, withDimension: 1, name: "ResNet.fcw:transposed")
        
        let fc = self.Linear(graph: graph, input: avgpoolReshaped, weight: fcwTransposed, bias: fcb, name: "ResNet.fc")
        return fc
    }
    
    func ResNet_S2_PreLayer3(graph: MPSGraph, input: MPSGraphTensor, params: [MPSGraphTensor], frozenRunningParams: Bool = true) -> MPSGraphTensor {
        let C = CounterFrom1()
        C.value = -1
        func nxt() -> MPSGraphTensor {
            return params[C.inc()]
        }
        return self.ResNet_S2_PreLayer3(graph: graph, input: input,
                                  l3_b0_c1w: nxt(), l3_b0_bn1w: nxt(), l3_b0_bn1b: nxt(), l3_b0_bn1rm: nxt(), l3_b0_bn1rv: nxt(),
                                  l3_b0_c2w: nxt(), l3_b0_bn2w: nxt(), l3_b0_bn2b: nxt(), l3_b0_bn2rm: nxt(), l3_b0_bn2rv: nxt(),
                                  l3_b0_ds_cw: nxt(), l3_b0_ds_bnw: nxt(), l3_b0_ds_bnb: nxt(), l3_b0_ds_bnrm: nxt(), l3_b0_ds_bnrv: nxt(),
                                  // block 1
                                  l3_b1_c1w: nxt(), l3_b1_bn1w: nxt(), l3_b1_bn1b: nxt(), l3_b1_bn1rm: nxt(), l3_b1_bn1rv: nxt(),
                                  l3_b1_c2w: nxt(), l3_b1_bn2w: nxt(), l3_b1_bn2b: nxt(), l3_b1_bn2rm: nxt(), l3_b1_bn2rv: nxt(),
                                  // block 2
                                  l3_b2_c1w: nxt(), l3_b2_bn1w: nxt(), l3_b2_bn1b: nxt(), l3_b2_bn1rm: nxt(), l3_b2_bn1rv: nxt(),
                                  l3_b2_c2w: nxt(), l3_b2_bn2w: nxt(), l3_b2_bn2b: nxt(), l3_b2_bn2rm: nxt(), l3_b2_bn2rv: nxt(),
                                  // block 3
                                  l3_b3_c1w: nxt(), l3_b3_bn1w: nxt(), l3_b3_bn1b: nxt(), l3_b3_bn1rm: nxt(), l3_b3_bn1rv: nxt(),
                                  l3_b3_c2w: nxt(), l3_b3_bn2w: nxt(), l3_b3_bn2b: nxt(), l3_b3_bn2rm: nxt(), l3_b3_bn2rv: nxt(),
                                  // block 4
                                  l3_b4_c1w: nxt(), l3_b4_bn1w: nxt(), l3_b4_bn1b: nxt(), l3_b4_bn1rm: nxt(), l3_b4_bn1rv: nxt(),
                                  l3_b4_c2w: nxt(), l3_b4_bn2w: nxt(), l3_b4_bn2b: nxt(), l3_b4_bn2rm: nxt(), l3_b4_bn2rv: nxt(),
                                  // block 5
                                  l3_b5_c1w: nxt(), l3_b5_bn1w: nxt(), l3_b5_bn1b: nxt(), l3_b5_bn1rm: nxt(), l3_b5_bn1rv: nxt(),
                                  l3_b5_c2w: nxt(), l3_b5_bn2w: nxt(), l3_b5_bn2b: nxt(), l3_b5_bn2rm: nxt(), l3_b5_bn2rv: nxt(),
                                         
                                 l4_b0_c1w: nxt(), l4_b0_bn1w: nxt(), l4_b0_bn1b: nxt(), l4_b0_bn1rm: nxt(), l4_b0_bn1rv: nxt(),
                                 l4_b0_c2w: nxt(), l4_b0_bn2w: nxt(), l4_b0_bn2b: nxt(), l4_b0_bn2rm: nxt(), l4_b0_bn2rv: nxt(),
                                 l4_b0_ds_cw: nxt(), l4_b0_ds_bnw: nxt(), l4_b0_ds_bnb: nxt(), l4_b0_ds_bnrm: nxt(), l4_b0_ds_bnrv: nxt(),
                                 l4_b1_c1w: nxt(), l4_b1_bn1w: nxt(), l4_b1_bn1b: nxt(), l4_b1_bn1rm: nxt(), l4_b1_bn1rv: nxt(),
                                 l4_b1_c2w: nxt(), l4_b1_bn2w: nxt(), l4_b1_bn2b: nxt(), l4_b1_bn2rm: nxt(), l4_b1_bn2rv: nxt(),
                                 l4_b2_c1w: nxt(), l4_b2_bn1w: nxt(), l4_b2_bn1b: nxt(), l4_b2_bn1rm: nxt(), l4_b2_bn1rv: nxt(),
                                 l4_b2_c2w: nxt(), l4_b2_bn2w: nxt(), l4_b2_bn2b: nxt(), l4_b2_bn2rm: nxt(), l4_b2_bn2rv: nxt(),
                                 fcw: nxt(), fcb: nxt(),
                                 frozenRunningParams: frozenRunningParams)
    }
    
        
    func ResNet_S2_PreLayer4(graph: MPSGraph, input: MPSGraphTensor,
                       // Only layer4+end
                          l4_b0_c1w: MPSGraphTensor, l4_b0_bn1w: MPSGraphTensor, l4_b0_bn1b: MPSGraphTensor, l4_b0_bn1rm: MPSGraphTensor, l4_b0_bn1rv: MPSGraphTensor,
                          l4_b0_c2w: MPSGraphTensor, l4_b0_bn2w: MPSGraphTensor, l4_b0_bn2b: MPSGraphTensor, l4_b0_bn2rm: MPSGraphTensor, l4_b0_bn2rv: MPSGraphTensor,
                          l4_b0_ds_cw: MPSGraphTensor, l4_b0_ds_bnw: MPSGraphTensor, l4_b0_ds_bnb: MPSGraphTensor, l4_b0_ds_bnrm: MPSGraphTensor, l4_b0_ds_bnrv: MPSGraphTensor,
                          l4_b1_c1w: MPSGraphTensor, l4_b1_bn1w: MPSGraphTensor, l4_b1_bn1b: MPSGraphTensor, l4_b1_bn1rm: MPSGraphTensor, l4_b1_bn1rv: MPSGraphTensor,
                          l4_b1_c2w: MPSGraphTensor, l4_b1_bn2w: MPSGraphTensor, l4_b1_bn2b: MPSGraphTensor, l4_b1_bn2rm: MPSGraphTensor, l4_b1_bn2rv: MPSGraphTensor,
                          l4_b2_c1w: MPSGraphTensor, l4_b2_bn1w: MPSGraphTensor, l4_b2_bn1b: MPSGraphTensor, l4_b2_bn1rm: MPSGraphTensor, l4_b2_bn1rv: MPSGraphTensor,
                          l4_b2_c2w: MPSGraphTensor, l4_b2_bn2w: MPSGraphTensor, l4_b2_bn2b: MPSGraphTensor, l4_b2_bn2rm: MPSGraphTensor, l4_b2_bn2rv: MPSGraphTensor,
                          
                          fcw: MPSGraphTensor, fcb: MPSGraphTensor,
                          
                          frozenRunningParams: Bool) -> MPSGraphTensor {
        // === Layer 4 ===
        let l4_b0 = self.ResNet_BasicBlockWithDownsample(graph: graph, input: input, c1weight: l4_b0_c1w, bn1weight: l4_b0_bn1w, bn1bias: l4_b0_bn1b,
                                                         bn1rm: l4_b0_bn1rm, bn1rv: l4_b0_bn1rv,
                                                         c2weight: l4_b0_c2w, bn2weight: l4_b0_bn2w, bn2bias: l4_b0_bn2b,
                                                         bn2rm: l4_b0_bn2rm, bn2rv: l4_b0_bn2rv,
                                                         ds_cw: l4_b0_ds_cw, ds_bnw: l4_b0_ds_bnw, ds_bnb: l4_b0_ds_bnb,
                                                         ds_bnrm: l4_b0_ds_bnrm, ds_bnrv: l4_b0_ds_bnrv,
                                                         name: "ResNet.layer4[0]", frozenRunningParams: frozenRunningParams)
        let l4_b1 = self.ResNet_BasicBlock(graph: graph, input: l4_b0, c1weight: l4_b1_c1w, bn1weight: l4_b1_bn1w, bn1bias: l4_b1_bn1b,
                                           bn1rm: l4_b1_bn1rm, bn1rv: l4_b1_bn1rv,
                                           c2weight: l4_b1_c2w, bn2weight: l4_b1_bn2w, bn2bias: l4_b1_bn2b,
                                           bn2rm: l4_b1_bn2rm, bn2rv: l4_b1_bn2rv,
                                           name: "ResNet.layer4[1]", frozenRunningParams: frozenRunningParams)
        let l4_b2 = self.ResNet_BasicBlock(graph: graph, input: l4_b1, c1weight: l4_b2_c1w, bn1weight: l4_b2_bn1w, bn1bias: l4_b2_bn1b,
                                           bn1rm: l4_b2_bn1rm, bn1rv: l4_b2_bn1rv,
                                           c2weight: l4_b2_c2w, bn2weight: l4_b2_bn2w, bn2bias: l4_b2_bn2b,
                                           bn2rm: l4_b2_bn2rm, bn2rv: l4_b2_bn2rv,
                                           name: "ResNet.layer4[2]", frozenRunningParams: frozenRunningParams)
        
        let avgpool = self.AdaptiveAvgPool2D1x1(graph: graph, input: l4_b2, name: "ResNet.avgpool")
        
        let avgpoolReshaped = graph.reshape(avgpool, shape: [avgpool.shape![0], avgpool.shape![1]], name: "ResNet.avgpool:reshapeForFC")
        let fcwTransposed = graph.transposeTensor(fcw, dimension: 0, withDimension: 1, name: "ResNet.fcw:transposed")
        
        let fc = self.Linear(graph: graph, input: avgpoolReshaped, weight: fcwTransposed, bias: fcb, name: "ResNet.fc")
        return fc
    }
        
    func ResNet_S2_PreLayer4(graph: MPSGraph, input: MPSGraphTensor, params: [MPSGraphTensor], frozenRunningParams: Bool = true) -> MPSGraphTensor {
        let C = CounterFrom1()
        C.value = -1
        func nxt() -> MPSGraphTensor {
            return params[C.inc()]
        }
        return self.ResNet_S2_PreLayer4(graph: graph, input: input,
                                 l4_b0_c1w: nxt(), l4_b0_bn1w: nxt(), l4_b0_bn1b: nxt(), l4_b0_bn1rm: nxt(), l4_b0_bn1rv: nxt(),
                                 l4_b0_c2w: nxt(), l4_b0_bn2w: nxt(), l4_b0_bn2b: nxt(), l4_b0_bn2rm: nxt(), l4_b0_bn2rv: nxt(),
                                 l4_b0_ds_cw: nxt(), l4_b0_ds_bnw: nxt(), l4_b0_ds_bnb: nxt(), l4_b0_ds_bnrm: nxt(), l4_b0_ds_bnrv: nxt(),
                                 l4_b1_c1w: nxt(), l4_b1_bn1w: nxt(), l4_b1_bn1b: nxt(), l4_b1_bn1rm: nxt(), l4_b1_bn1rv: nxt(),
                                 l4_b1_c2w: nxt(), l4_b1_bn2w: nxt(), l4_b1_bn2b: nxt(), l4_b1_bn2rm: nxt(), l4_b1_bn2rv: nxt(),
                                 l4_b2_c1w: nxt(), l4_b2_bn1w: nxt(), l4_b2_bn1b: nxt(), l4_b2_bn1rm: nxt(), l4_b2_bn1rv: nxt(),
                                 l4_b2_c2w: nxt(), l4_b2_bn2w: nxt(), l4_b2_bn2b: nxt(), l4_b2_bn2rm: nxt(), l4_b2_bn2rv: nxt(),
                                 fcw: nxt(), fcb: nxt(),
                                 frozenRunningParams: frozenRunningParams)
    }
}


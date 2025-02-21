use std::fs::File;
use std::usize;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
use crate::operators::rms_norm;
use crate::operators::matmul_transb;
use crate::operators::swiglu;
use crate::operators::masked_softmax;

use rand::prelude::*;
use rand::distributions::WeightedIndex; 
use ndarray::s; 
// use tch::Tensor;

use std::ops::AddAssign;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> 
{
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            // todo!("self_attention(...)");
            // todo!("down_proj matmul and add residual");

            // todo!("mlp(...)");

            // -------new starts here-----------
            
            // --- 4. 自注意力 ---
            self_attention(
                &mut hidden_states, // 用来盛放注意力结果
                &mut att_scores,
                &q_buf,   // Q
                full_k,   // K (含历史tokens)
                full_v,   // V (含历史tokens)
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );
            
            // --- 5. 下游投影 (wo) + 残差 ---
            // 这一部分把 hidden_states 投影回 d 维度，然后加到 residual 上
            let mut att_output = Tensor::<f32>::default(&vec![seq_len, self.d]);
            OP::matmul_transb(&mut att_output, 0.0, &hidden_states, &self.params.wo[layer], 1.0);

            for i in 0..residual.len() {
                residual[i] += att_output[i];
            }


            // --- 6. MLP ---
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );


        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }


pub fn apply_temperature(logits: &mut Vec<f32>, temperature: f32) {
    // 0 for greedy, 1 for non-resize
    if temperature <= 0.0 {
        return;
    }
    let inv_temp = 1.0 / temperature;
    for logit in logits.iter_mut() {
        *logit *= inv_temp;
    }
}
fn top_k_filter(probs:&mut [f32],k:usize)
{
    //要选top k个 但是没有k个给我们选
    if k >= probs.len()
    {
        return;
    }
    // 找到第 k 大的元素（这一块咨询了gpt
    let mut probs_copy = probs.to_vec(); // 复制一份
    let (_, kth_largest, _) = probs_copy.select_nth_unstable_by(k, |a, b| b.partial_cmp(a).unwrap());

    let threshold = *kth_largest; // k 大元素中的最小值

    // 过滤掉小于 threshold 的元素
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }

}

//概率从大到小累加，直到超过p的部分 为0
pub fn top_p_filter(probs:&mut[f32],p:f32)
{
    // 1. 复制并排序，保留原索引
    let mut indexed_probs: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // 降序排序

    // 2. 计算累积概率
    let mut cumulative_prob = 0.0;
    let mut threshold_index = indexed_probs.len(); // 记录需要保留的范围
    for (i, &(_, prob)) in indexed_probs.iter().enumerate() {
        cumulative_prob += prob;
        if cumulative_prob >= p {
            threshold_index = i + 1; // 保留前 i+1 个元素
            break;
        }
    }

    // 3. 过滤掉累积概率之外的值
    let threshold_set: std::collections::HashSet<usize> = indexed_probs[..threshold_index]
        .iter()
        .map(|&(idx, _)| idx)
        .collect();

    for (i, prob) in probs.iter_mut().enumerate() {
        if !threshold_set.contains(&i) {
            *prob = 0.0;
        }
    }
}



// pub fn sample_from_logits(
//     logits:&Tensor<f32>,
//     top_p:f32,
//     top_k:usize,
//     temperature: f32,
//     rng:&mut ThreadRng,
// )-> u32{
//     let mut probs = logits.data().to_vec();

//     apply_temperature(&mut probs,temperature);
// }
pub fn sample_from_logits(
    logits: &Tensor<f32>,
    top_p: f32,
    top_k: usize,
    temperature: f32,
    rng: &mut ThreadRng,
) -> u32 {
    let mut probs = logits.data().to_vec(); // -> Vec<f32>

// 显式转成切片
    Self::apply_temperature(&mut probs, temperature);
    Self::top_k_filter(&mut probs, top_k);
    Self::top_p_filter(probs.as_mut_slice(), top_p);

    // 3. 归一化到概率分布
    let sum: f32 = probs.iter().sum();
    if sum > 1e-8 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    } else {
        // 如果数值异常，可以做一个fallback，这里简单给 uniform
        let val = 1.0 / (probs.len() as f32);
        for p in probs.iter_mut() {
            *p = val;
        }
    }

    // 4. multinomial 抽样
    let dist = WeightedIndex::new(&probs).unwrap();
    dist.sample(rng) as u32
}


    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32>{
        // let mut result = Vec::<u32>::new();
        
        // // todo!("实现文本生成");
        // let mut rng= thread_rng();
        // let mut result = token_ids.to_vec();
        // let mut cache = self.new_cache();
        // for _ in 0..max_len
        // {
        //     let input_tensor = Tensor::<u32>::new(result.clone(),&vec![result.len()]);
        //     let logits = self.forward(&input_tensor, &mut cache);
        //     // let next_token= sample_from_logits(&logits,top_p,top_k,)
        // }
        // result

        

    }
}

// fn self_attention(
//     hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
//     att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
//     q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
//     k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
//     v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
//     n_kv_h: usize,
//     n_groups: usize,
//     seq_len: usize,
//     total_seq_len: usize,
//     dqkv: usize,
// ) 
// {
//     // todo!("Implement self_attention");
//     // 1. 计算 Q K^T, 存入 att_scores
//     matmul_transb(att_scores, 0.0, q, k, 1.0);

//     // 2. 归一化 att_scores /= sqrt(dim)
//     let scale = 1.0 / (dqkv as f32).sqrt();
//     let attn_data = unsafe { att_scores.data_mut() };
//     for v in attn_data.iter_mut() {
//         *v *= scale;
//     }

//     // 3. 计算 softmax
//     masked_softmax(att_scores);

//     // 4. 计算 attn_V = attn @ V
//     let mut attn_v = Tensor::<f32>::zeros(hidden_states.shape());
//     matmul_transb(&mut attn_v, 0.0, att_scores, v, 1.0);

//     // 5. 更新 hidden_states (手动逐元素相加)
//     let hidden_data = unsafe { hidden_states.data_mut() };
//     let attn_data = attn_v.data(); // 只读数据

//     for (h, a) in hidden_data.iter_mut().zip(attn_data.iter()) 
//     {
//         *h += a;
//     }
    
// }


fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_q_h*dqkv) == (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_q_h*dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h*dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h*dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) 
{
    // todo!("Implement self_attention");
    // 1. 计算 Q K^T, 存入 att_scores
    matmul_transb(att_scores, 0.0, q, k, 1.0);

    // 2. 缩放
    let scale = 1.0 / (head_dim as f32).sqrt();
    for v in scores_data.iter_mut() {
        *v *= scale;
    }

    // 3. masked_softmax
    //    这里假设你实现了一个 4D 版本的 masked_softmax，如果没有掩码就直接普通 softmax 即可
    masked_softmax(att_scores);

    // 4. att_scores * V -> hidden_states
    //    hidden_states 原 shape (seq, n_q_h*dqkv)
    //    这里需要把每个 head 的注意力得分乘以对应的 V，然后再 sum 到一起
    let v_data = v.data(); 
    let mut out_data = vec![0.0; hidden_states.len()]; // 收集计算结果

    for (h, a) in hidden_data.iter_mut().zip(attn_data.iter()) 
    {
        *h += a;
    }

    // 把 out_data 拷回 hidden_states
    let hs_data = unsafe { hidden_states.data_mut() };
    for i in 0..hs_data.len() {
        hs_data[i] = out_data[i];
    }
}
fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
     // 1. 对 residual 做 RMSNorm，结果放到 hidden_states 中
    rms_norm(hidden_states, residual, rms_w, eps);

    // 2. 计算 gate = hidden_states @ w_gate.T
    matmul_transb(gate, 0.0, hidden_states, w_gate, 1.0);

    // 3. 计算 up = hidden_states @ w_up.T
    matmul_transb(up, 0.0, hidden_states, w_up, 1.0);


    for i in 0..gate.len() {
        let g = gate[i];
        gate[i] = g * sigmoid(g) * up[i];
    }

    // 5. 计算输出 hidden_states = gate @ w_down.T
    matmul_transb(hidden_states, 0.0, gate, w_down, 1.0);

    // 6. 残差连接：residual += hidden_states
    //    然后把新的结果拷回 hidden_states 供后面使用
    for i in 0..residual.len() {
        residual[i] += hidden_states[i];
    }
    hidden_states.copy_from(residual);

}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}

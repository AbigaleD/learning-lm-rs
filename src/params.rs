use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use bytemuck::cast_slice;
use safetensors::tensor::TensorView;

fn get_tensor(safetensor: &SafeTensors, name: &str) -> Tensor<f32> {
    let tv: TensorView = safetensor
        .tensor(name)
        .unwrap_or_else(|_| panic!("Tensor '{}' not found in safetensors", name));
    let shape = tv.shape().to_vec();
    let raw_data = tv.data();
    let data_f32: &[f32] = cast_slice(raw_data);
    let data_vec = data_f32.to_vec();
    Tensor::new(data_vec, &shape)
}

fn find_num_layers(safetensor: &SafeTensors) -> usize {
    let mut count = 0;
    loop {
        let key = format!("model.layers.{count}.input_layernorm.weight");
        if safetensor.tensor(&key).is_ok() {
            count += 1;
        } else {
            return count;
        }
    }
    // count
}

pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // ...
    pub wo: Vec<Tensor<T>>,
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>,
    pub w_up: Vec<Tensor<T>>,
    pub w_gate: Vec<Tensor<T>>,
    pub w_down: Vec<Tensor<T>>,
    // output
    pub rms_out_w: Tensor<T>,
    pub lm_head: Tensor<T>,
}

impl LLamaParams<f32> {
    /// 从 safetensors 文件加载各层权重
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // 1. 如果 config 里没有显式指定层数，就用 find_num_layers 探测
        let n_layers = if config.num_hidden_layers > 0 {
            config.num_hidden_layers as usize
        } else {
            find_num_layers(safetensor)
        };

        // 2. 加载“embedding”。如果文件里没有 model.embed_tokens.weight，就
        //    用 lm_head.weight 作为 embedding_table（有些模型就是 tie embeddings）
        let embedding_table = match safetensor.tensor("model.embed_tokens.weight") {
            Ok(_) => get_tensor(safetensor, "model.embed_tokens.weight"),
            Err(_) => {
                eprintln!("WARNING: 'model.embed_tokens.weight' not found. Fallback => 'lm_head.weight' as embedding table.");
                get_tensor(safetensor, "lm_head.weight")
            }
        };

        // 3. 先准备容器
        let mut rms_att_w = Vec::with_capacity(n_layers);
        let mut wq = Vec::with_capacity(n_layers);
        let mut wk = Vec::with_capacity(n_layers);
        let mut wv = Vec::with_capacity(n_layers);
        let mut wo = Vec::with_capacity(n_layers);

        let mut rms_ffn_w = Vec::with_capacity(n_layers);
        let mut w_up = Vec::with_capacity(n_layers);
        let mut w_gate = Vec::with_capacity(n_layers);
        let mut w_down = Vec::with_capacity(n_layers);

        // 4. 循环加载每层
        // 注意：下面的命名必须与你 safetensors 文件实际保存的一致
        for i in 0..n_layers {
            // 原先 attention_norm.weight => 改为 input_layernorm.weight
            rms_att_w.push(
                get_tensor(safetensor, &format!("model.layers.{i}.input_layernorm.weight"))
            );

            // 原先 attention.wq.weight => self_attn.q_proj.weight
            wq.push(get_tensor(safetensor, &format!("model.layers.{i}.self_attn.q_proj.weight")));
            wk.push(get_tensor(safetensor, &format!("model.layers.{i}.self_attn.k_proj.weight")));
            wv.push(get_tensor(safetensor, &format!("model.layers.{i}.self_attn.v_proj.weight")));
            wo.push(get_tensor(safetensor, &format!("model.layers.{i}.self_attn.o_proj.weight")));

            // 原先 ffn_norm.weight => post_attention_layernorm.weight
            rms_ffn_w.push(
                get_tensor(safetensor, &format!("model.layers.{i}.post_attention_layernorm.weight"))
            );
            // feed_forward.w_up => mlp.up_proj
            w_up.push(get_tensor(safetensor, &format!("model.layers.{i}.mlp.up_proj.weight")));
            w_gate.push(get_tensor(safetensor, &format!("model.layers.{i}.mlp.gate_proj.weight")));
            w_down.push(get_tensor(safetensor, &format!("model.layers.{i}.mlp.down_proj.weight")));
        }

        let rms_out_w = get_tensor(safetensor, "model.norm.weight");

        let lm_head = get_tensor(safetensor, "lm_head.weight");

        LLamaParams {
            embedding_table,
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w,
            lm_head,
        }
    }
}
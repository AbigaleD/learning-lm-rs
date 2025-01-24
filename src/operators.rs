use crate::tensor::Tensor;
// 从表中获取行向量（gather）
// get (row) vectors from a 2D table given a list of indices
// 	1.	输入参数：
// •	y：目标张量，用于存储提取出的行向量。
// •	indices：索引张量，表示需要从表中提取的行号。
// •	table：源二维表
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
// 	1.	输入参数：
// •	y：需要进行旋转嵌入的张量，形状为[seq_len, n_heads, d]。
// •	start_pos：序列的起始位置。
// •	theta：控制频率的参数。
// 2.	关键逻辑：
// •	确保输入张量是三维的，提取序列长度（seq_len）、注意力头数（n_heads）和维度（d）。
// •	遍历序列中的每个token，计算对应的频率（freq），并根据正弦和余弦旋转嵌入更新y
// 核心公式：
// •	更新公式：

// \text{data}[i] = a \cdot \cos(\text{freq}) - b \cdot \sin(\text{freq})


// \text{data}[i + d/2] = b \cdot \cos(\text{freq}) + a \cdot \sin(\text{freq})


// 实现细节：
//     •	使用theta.powf((i * 2) as f32 / d as f32)计算频率，确保维度间频率逐渐增加。
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
// 3. masked_softmax函数

// 作用：计算具有掩码的softmax，常用于处理注意力机制中的不规则序列。

// 代码解析：
// 	1.	输入参数：
// 	•	y：输入张量，最后两维表示[seq_len, total_seq_len]。
// 	2.	关键逻辑：
// 	•	遍历每个batch和每个序列，计算softmax。
// 	•	在计算softmax时：
// 	•	找到当前范围内的最大值进行数值稳定化。
// 	•	计算指数值并归一化为概率。
// 	•	对超出掩码范围的部分置零。
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    // assert!(
    //     x.shape() == y.shape(),
    //     "Shapes of x and y must be equal. x: {:?}, y: {:?}",
    //     x.shape(),
    //     y.shape()
    // );
    assert!(
        w.shape().len() == 1 && *w.shape() == vec![*x.shape().last().unwrap()],
        "Shape of w must match the last dimension of x. w: {:?}, x_last_dim: {:?}",
        w.shape(),
        x.shape().last()
    );

   
    let batch_size = x.shape()[..x.shape().len() - 1].iter().product::<usize>(); // 批大小
    let dim = *x.shape().last().unwrap(); // 最后一维的长度

    // 获取数据切片
    let x_data = x.data();
    let w_data = w.data();
    let y_data = unsafe { y.data_mut() };

    // 遍历每个批次进行计算
    for i in 0..batch_size {
        let start = i * dim;
        let x_batch = &x_data[start..start + dim];
        let y_batch = &mut y_data[start..start + dim];

        // 计算当前批次的 RMS
        let rms = (x_batch.iter().map(|&v| v * v).sum::<f32>() / dim as f32 + epsilon).sqrt();

        for j in 0..dim {
            y_batch[j] = w_data[j] * x_batch[j] / rms;
        }
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    for i in 0..len {
        _y[i] *= _x[i] / (1. + (-_x[i]).exp());
    }
    // todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    
    assert_eq!(a.shape().len(), 2, "A 必须是 2 维张量");
    assert_eq!(b.shape().len(), 2, "B 必须是 2 维张量");
    assert_eq!(c.shape().len(), 2, "C 必须是 2 维张量");


    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[0];
    


    debug_assert_eq!(k, b.shape()[1]);
    debug_assert_eq!(c.shape()[0], m);
    debug_assert_eq!(c.shape()[1], n);

    let a_data = a.data();        // A只读
    let b_data = b.data();        // B只读
    let c_data = unsafe { c.data_mut() }; // C要写

    //    c[i, j] = beta * c[i, j] + alpha * Σ_p( A[i, p] * B[j, p] )
    for i in 0..m {
        for j in 0..n {
            // 先乘 beta
            let mut tmp = beta * c_data[i * n + j];
            // 再加上 alpha * ( A[i,p] * B[j,p] ) 的累加和
            let mut sum = 0.0;
            for p in 0..k {
                sum += a_data[i * k + p] * b_data[j * k + p];
            }
            // 写回 C
            c_data[i * n + j] = tmp + alpha * sum;
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}

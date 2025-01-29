use std::{slice, sync::Arc, vec};
use std::ops::{Index, IndexMut};
pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Copy + Clone + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length: length,
        }
    }
// •	.product() 计算所有元素的乘积，用于求出张量中所有元素的数量。
// •	例如：
// •	对于 shape = vec![2, 3]，length = 2 * 3 = 6。
// •	对于 shape = vec![3, 3, 3]，length = 3 * 3 * 3 = 27。
    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }
    pub fn clone(&self) -> Self {
        // Arc<Box<[T]>> 调用 .clone() 会做引用计数的浅复制
        // shape.clone() 会复制一个新的 Vec<usize>
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            offset: self.offset,
            length: self.length,
        }
    }


}
impl Index<usize> for Tensor<f32> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        // 注意：这里的 self.data() 已经是切片 &self.data[self.offset..][..self.length]
        // 所以外部看到的 [0..length] 就是张量有效部分
        &self.data()[index]
    }
}

impl IndexMut<usize> for Tensor<f32> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // data_mut() 是 unsafe 的，因为涉及到从 Arc<Box<[T]>> 中取出可变指针。
        // 这里我们直接用 unsafe 包裹
        unsafe {
            &mut self.data_mut()[index]
        }
    }
}
// Some helper functions for testing and debugging
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();
        
        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self){
        println!("shpae: {:?}, offset: {}, length: {}", self.shape, self.offset, self.length);
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
    #[allow(unused)]
    pub fn copy_from(&mut self, other: &Self) {
        if self.size() != other.size() {
            panic!(
                "copy_from: size mismatch, self.size()={}, other.size()={}",
                self.size(),
                other.size()
            );
        }
        unsafe {
            self.data_mut().copy_from_slice(other.data());
        }
    }
    #[allow(unused)]
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}

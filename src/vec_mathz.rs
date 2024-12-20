//! 三维向量数学库
//! 
//! 提供了三维向量的基本运算和几何操作功能

mod ops;
mod matrix;
mod matrix_mn;

pub use ops::*;
pub use matrix::Mat3;
pub use matrix_mn::MatMN;

/// 三维向量结构
/// 
/// 包含x、y、z三个分量的浮点数向量
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Default for Vec3 {
    fn default() -> Self {
        Self::zero()
    }
}

// 基本构造方法
impl Vec3 {
    /// 创建新的三维向量
    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    /// 创建零向量
    #[inline]
    pub fn zero() -> Self {
        Vec3::new(0.0, 0.0, 0.0)
    }

    /// 创建x轴单位向量
    #[inline]
    pub fn unit_x() -> Self {
        Vec3::new(1.0, 0.0, 0.0)
    }

    /// 创建y轴单位向量
    #[inline]
    pub fn unit_y() -> Self {
        Vec3::new(0.0, 1.0, 0.0)
    }

    /// 创建z轴单位向量
    #[inline]
    pub fn unit_z() -> Self {
        Vec3::new(0.0, 0.0, 1.0)
    }

    /// 从数组创建向量
    #[inline]
    pub fn from_array(arr: [f64; 3]) -> Self {
        Vec3::new(arr[0], arr[1], arr[2])
    }

    /// 转换为数组
    #[inline]
    pub fn to_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

// 基本运算方法
impl Vec3 {
    /// 计算向量长度
    #[inline]
    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    /// 计算向量长度的平方
    #[inline]
    pub fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// 向量归一化
    #[inline]
    pub fn normalize(&self) -> Vec3 {
        let len = self.length();
        if len > 0.0 {
            *self / len
        } else {
            *self
        }
    }

    /// 判断是否为零向量
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.length_squared() < 1e-10
    }

    /// 限制向量长度
    #[inline]
    pub fn clamp_length(&self, max_length: f64) -> Vec3 {
        let length = self.length();
        if length > max_length {
            *self * (max_length / length)
        } else {
            *self
        }
    }
}

// 向量代数运算
impl Vec3 {
    /// 计算点积
    #[inline]
    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// 计算叉积
    #[inline]
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// 计算向量在另一个向量上的投影
    #[inline]
    pub fn project_onto(&self, other: &Vec3) -> Vec3 {
        let other_normalized = other.normalize();
        other_normalized * self.dot(&other_normalized)
    }

    /// 计算向量关于法向量的反射
    #[inline]
    pub fn reflect(&self, normal: &Vec3) -> Vec3 {
        *self - *normal * 2.0 * self.dot(normal)
    }
}

// 几何相关方法
impl Vec3 {
    /// 计算与另一个向量的夹角（弧度）
    #[inline]
    pub fn angle(&self, other: &Vec3) -> f64 {
        let cos_theta = self.dot(other) / (self.length() * other.length());
        cos_theta.clamp(-1.0, 1.0).acos()
    }

    /// 计算与另一个向量的距离
    #[inline]
    pub fn distance(&self, other: &Vec3) -> f64 {
        (*self - *other).length()
    }

    /// 计算与另一个向量的中点
    #[inline]
    pub fn midpoint(&self, other: &Vec3) -> Vec3 {
        (*self + *other) * 0.5
    }

    /// 判断是否与另一个向量平行
    #[inline]
    pub fn is_parallel(&self, other: &Vec3) -> bool {
        self.cross(other).length_squared() < 1e-10
    }

    /// 判断是否与另一个向量垂直
    #[inline]
    pub fn is_perpendicular(&self, other: &Vec3) -> bool {
        self.dot(other).abs() < 1e-10
    }
}

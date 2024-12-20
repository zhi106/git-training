//! 矩阵运算模块
//! 
//! 提供3x3矩阵的基本运算和变换功能

use super::Vec3;
use std::ops::*;

/// 3x3矩阵结构
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    /// 矩阵数据，按行存储
    data: [[f64; 3]; 3],
}

impl Default for Mat3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Mat3 {
    /// 创建新的3x3矩阵
    #[inline]
    pub fn new(data: [[f64; 3]; 3]) -> Self {
        Mat3 { data }
    }

    /// 创建单位矩阵
    #[inline]
    pub fn identity() -> Self {
        Mat3::new([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    }

    /// 创建零矩阵
    #[inline]
    pub fn zero() -> Self {
        Mat3::new([[0.0; 3]; 3])
    }

    /// 从列向量创建矩阵
    #[inline]
    pub fn from_cols(col1: Vec3, col2: Vec3, col3: Vec3) -> Self {
        Mat3::new([
            [col1.x, col2.x, col3.x],
            [col1.y, col2.y, col3.y],
            [col1.z, col2.z, col3.z],
        ])
    }

    /// 从行向量创建矩阵
    #[inline]
    pub fn from_rows(row1: Vec3, row2: Vec3, row3: Vec3) -> Self {
        Mat3::new([
            [row1.x, row1.y, row1.z],
            [row2.x, row2.y, row2.z],
            [row3.x, row3.y, row3.z],
        ])
    }

    /// 获取指定位置的元素
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row][col]
    }

    /// 设置指定位置的元素
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row][col] = value;
    }

    /// 获取指定列
    #[inline]
    pub fn col(&self, col: usize) -> Vec3 {
        Vec3::new(
            self.data[0][col],
            self.data[1][col],
            self.data[2][col],
        )
    }

    /// 获取指定行
    #[inline]
    pub fn row(&self, row: usize) -> Vec3 {
        Vec3::new(
            self.data[row][0],
            self.data[row][1],
            self.data[row][2],
        )
    }

    /// 计算矩阵的转置
    #[inline]
    pub fn transpose(&self) -> Mat3 {
        let mut result = Mat3::zero();
        for i in 0..3 {
            for j in 0..3 {
                result.data[i][j] = self.data[j][i];
            }
        }
        result
    }

    /// 计算矩阵的行列式
    #[inline]
    pub fn determinant(&self) -> f64 {
        let [[a, b, c], [d, e, f], [g, h, i]] = self.data;
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }

    /// 计算矩阵的逆
    /// 如果矩阵不可逆，返回None
    pub fn inverse(&self) -> Option<Mat3> {
        let det = self.determinant();
        if det.abs() < 1e-10 {
            return None;
        }

        let [[a, b, c], [d, e, f], [g, h, i]] = self.data;
        let adj = Mat3::new([
            [e*i - f*h, c*h - b*i, b*f - c*e],
            [f*g - d*i, a*i - c*g, c*d - a*f],
            [d*h - e*g, b*g - a*h, a*e - b*d],
        ]);

        Some(adj * (1.0 / det))
    }

    /// 创建旋转矩阵（绕x轴旋转）
    #[inline]
    pub fn rotation_x(angle: f64) -> Self {
        let (sin, cos) = angle.sin_cos();
        Mat3::new([
            [1.0, 0.0,  0.0],
            [0.0, cos, -sin],
            [0.0, sin,  cos],
        ])
    }

    /// 创建旋转矩阵（绕y轴旋转）
    #[inline]
    pub fn rotation_y(angle: f64) -> Self {
        let (sin, cos) = angle.sin_cos();
        Mat3::new([
            [ cos, 0.0, sin],
            [ 0.0, 1.0, 0.0],
            [-sin, 0.0, cos],
        ])
    }

    /// 创建旋转矩阵（绕z轴旋转）
    #[inline]
    pub fn rotation_z(angle: f64) -> Self {
        let (sin, cos) = angle.sin_cos();
        Mat3::new([
            [cos, -sin, 0.0],
            [sin,  cos, 0.0],
            [0.0,  0.0, 1.0],
        ])
    }

    /// 创建缩放矩阵
    #[inline]
    pub fn scaling(sx: f64, sy: f64, sz: f64) -> Self {
        Mat3::new([
            [sx,  0.0, 0.0],
            [0.0, sy,  0.0],
            [0.0, 0.0, sz ],
        ])
    }

    /// 创建错切矩阵（XY平面）
    #[inline]
    pub fn shear_xy(k_x: f64, k_y: f64) -> Self {
        Mat3::new([
            [1.0, k_x, 0.0],
            [k_y, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    }

    /// 创建反射矩阵（关于指定向量）
    pub fn reflection(normal: &Vec3) -> Self {
        let n = normal.normalize();
        let nx2 = n.x * n.x * 2.0;
        let ny2 = n.y * n.y * 2.0;
        let nz2 = n.z * n.z * 2.0;
        let nxy2 = n.x * n.y * 2.0;
        let nxz2 = n.x * n.z * 2.0;
        let nyz2 = n.y * n.z * 2.0;

        Mat3::new([
            [1.0 - nx2, -nxy2, -nxz2],
            [-nxy2, 1.0 - ny2, -nyz2],
            [-nxz2, -nyz2, 1.0 - nz2],
        ])
    }

    /// 创建投影矩阵（投影到指定向量）
    pub fn projection(normal: &Vec3) -> Self {
        let n = normal.normalize();
        let nx2 = n.x * n.x;
        let ny2 = n.y * n.y;
        let nz2 = n.z * n.z;
        let nxy = n.x * n.y;
        let nxz = n.x * n.z;
        let nyz = n.y * n.z;

        Mat3::new([
            [nx2, nxy, nxz],
            [nxy, ny2, nyz],
            [nxz, nyz, nz2],
        ])
    }
}

// 矩阵乘法
impl Mul for Mat3 {
    type Output = Mat3;

    #[inline]
    fn mul(self, rhs: Mat3) -> Self::Output {
        let mut result = Mat3::zero();
        for i in 0..3 {
            for j in 0..3 {
                result.data[i][j] = (0..3).map(|k| self.data[i][k] * rhs.data[k][j]).sum();
            }
        }
        result
    }
}

// 矩阵与向量相乘
impl Mul<Vec3> for Mat3 {
    type Output = Vec3;

    #[inline]
    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3::new(
            self.row(0).dot(&rhs),
            self.row(1).dot(&rhs),
            self.row(2).dot(&rhs),
        )
    }
}

// 矩阵与标量相乘
impl Mul<f64> for Mat3 {
    type Output = Mat3;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        let mut result = Mat3::zero();
        for i in 0..3 {
            for j in 0..3 {
                result.data[i][j] = self.data[i][j] * rhs;
            }
        }
        result
    }
}

// 标量与矩阵相乘
impl Mul<Mat3> for f64 {
    type Output = Mat3;

    #[inline]
    fn mul(self, rhs: Mat3) -> Self::Output {
        rhs * self
    }
}

// 矩阵加法
impl Add for Mat3 {
    type Output = Mat3;

    #[inline]
    fn add(self, rhs: Mat3) -> Self::Output {
        let mut result = Mat3::zero();
        for i in 0..3 {
            for j in 0..3 {
                result.data[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }
        result
    }
}

// 矩阵减法
impl Sub for Mat3 {
    type Output = Mat3;

    #[inline]
    fn sub(self, rhs: Mat3) -> Self::Output {
        let mut result = Mat3::zero();
        for i in 0..3 {
            for j in 0..3 {
                result.data[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }
        result
    }
}

// 矩阵取负
impl Neg for Mat3 {
    type Output = Mat3;

    #[inline]
    fn neg(self) -> Self::Output {
        let mut result = Mat3::zero();
        for i in 0..3 {
            for j in 0..3 {
                result.data[i][j] = -self.data[i][j];
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_matrix_constructors() {
        let m = Mat3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 1), 5.0);
        assert_eq!(m.get(2, 2), 9.0);

        let identity = Mat3::identity();
        assert_eq!(identity.get(0, 0), 1.0);
        assert_eq!(identity.get(1, 1), 1.0);
        assert_eq!(identity.get(2, 2), 1.0);
        assert_eq!(identity.get(0, 1), 0.0);

        let zero = Mat3::zero();
        assert_eq!(zero.get(0, 0), 0.0);
        assert_eq!(zero.get(1, 1), 0.0);
        assert_eq!(zero.get(2, 2), 0.0);
    }

    #[test]
    fn test_matrix_operations() {
        let m1 = Mat3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let m2 = Mat3::new([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);

        // 加法
        let sum = m1 + m2;
        assert_eq!(sum.get(0, 0), 10.0);
        assert_eq!(sum.get(1, 1), 10.0);
        assert_eq!(sum.get(2, 2), 10.0);

        // 减法
        let diff = m1 - m2;
        assert_eq!(diff.get(0, 0), -8.0);
        assert_eq!(diff.get(1, 1), 0.0);
        assert_eq!(diff.get(2, 2), 8.0);

        // 标量乘法
        let scaled = m1 * 2.0;
        assert_eq!(scaled.get(0, 0), 2.0);
        assert_eq!(scaled.get(1, 1), 10.0);
        assert_eq!(scaled.get(2, 2), 18.0);
    }

    #[test]
    fn test_matrix_multiplication() {
        let m1 = Mat3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let m2 = Mat3::new([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);

        let result = m1 * m2;
        assert_eq!(result.get(0, 0), 30.0);
        assert_eq!(result.get(1, 1), 81.0);
        assert_eq!(result.get(2, 2), 132.0);

        // 与单位矩阵相乘
        let identity = Mat3::identity();
        let result = m1 * identity;
        assert_eq!(result.data, m1.data);
    }

    #[test]
    fn test_matrix_vector_multiplication() {
        let m = Mat3::new([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
        let v = Vec3::new(1.0, 1.0, 1.0);

        let result = m * v;
        assert_eq!(result.x, 1.0);
        assert_eq!(result.y, 2.0);
        assert_eq!(result.z, 3.0);
    }

    #[test]
    fn test_matrix_transformations() {
        // 旋转矩阵测试
        let rot_x = Mat3::rotation_x(PI / 2.0);
        let v = Vec3::new(0.0, 1.0, 0.0);
        let rotated = rot_x * v;
        assert!((rotated.y + 1.0).abs() < 1e-10);
        assert!((rotated.z - 1.0).abs() < 1e-10);

        // 缩放矩阵测试
        let scale = Mat3::scaling(2.0, 3.0, 4.0);
        let v = Vec3::new(1.0, 1.0, 1.0);
        let scaled = scale * v;
        assert_eq!(scaled.x, 2.0);
        assert_eq!(scaled.y, 3.0);
        assert_eq!(scaled.z, 4.0);
    }

    #[test]
    fn test_matrix_inverse() {
        let m = Mat3::new([[4.0, 7.0, 2.0], [2.0, 6.0, 3.0], [1.0, 8.0, 5.0]]);
        let inv = m.inverse().unwrap();
        
        // 验证 M * M^(-1) = I
        let result = m * inv;
        let identity = Mat3::identity();
        for i in 0..3 {
            for j in 0..3 {
                assert!((result.get(i, j) - identity.get(i, j)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_linear_transformations() {
        // 测试错切变换
        let shear = Mat3::shear_xy(1.0, 0.0);
        let v = Vec3::new(1.0, 1.0, 1.0);
        let sheared = shear * v;
        assert_eq!(sheared.x, 2.0); // x + y
        assert_eq!(sheared.y, 1.0); // y
        assert_eq!(sheared.z, 1.0); // z

        // 测试反射变换
        let normal = Vec3::new(1.0, 0.0, 0.0); // 关于yz平面反射
        let reflection = Mat3::reflection(&normal);
        let v = Vec3::new(1.0, 1.0, 1.0);
        let reflected = reflection * v;
        assert!((reflected.x + 1.0).abs() < 1e-10);
        assert_eq!(reflected.y, 1.0);
        assert_eq!(reflected.z, 1.0);

        // 测试投影变换
        let normal = Vec3::new(0.0, 1.0, 0.0); // 投影到xz平面
        let projection = Mat3::projection(&normal);
        let v = Vec3::new(1.0, 1.0, 1.0);
        let projected = projection * v;
        assert_eq!(projected.x, 0.0);
        assert_eq!(projected.y, 1.0);
        assert_eq!(projected.z, 0.0);
    }
} 
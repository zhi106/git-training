//! 通用矩阵运算模块
//! 
//! 提供任意大小矩阵的基本运算和变换功能

use std::ops::*;
use std::fmt;

/// MxN矩阵结构
#[derive(Clone, PartialEq)]
pub struct MatMN {
    /// 矩阵数据，按行存储
    data: Vec<f64>,
    /// 行数
    rows: usize,
    /// 列数
    cols: usize,
}

impl fmt::Debug for MatMN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix {}x{}", self.rows, self.cols)?;
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:8.4}", self.get(i, j))?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

impl MatMN {
    /// 创建新的MxN矩阵
    pub fn new(rows: usize, cols: usize) -> Self {
        MatMN {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// 从二维数组创建矩阵
    pub fn from_array(data: &[&[f64]]) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        let mut mat = MatMN::new(rows, cols);
        
        for i in 0..rows {
            for j in 0..cols {
                mat.set(i, j, data[i][j]);
            }
        }
        mat
    }

    /// 创建单位矩阵
    pub fn identity(size: usize) -> Self {
        let mut mat = MatMN::new(size, size);
        for i in 0..size {
            mat.set(i, i, 1.0);
        }
        mat
    }

    /// 获取矩阵大小
    #[inline]
    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// 获取指定位置的元素
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    /// 设置指定位置的元素
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }

    /// 获取指定行
    pub fn get_row(&self, row: usize) -> Vec<f64> {
        let start = row * self.cols;
        self.data[start..start + self.cols].to_vec()
    }

    /// 获取指定列
    pub fn get_col(&self, col: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            result.push(self.get(i, col));
        }
        result
    }

    /// 计算矩阵的转置
    pub fn transpose(&self) -> MatMN {
        let mut result = MatMN::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// 计算矩阵的行列式（仅适用于方阵）
    pub fn determinant(&self) -> Option<f64> {
        if self.rows != self.cols {
            return None;
        }

        match self.rows {
            0 => Some(1.0),
            1 => Some(self.get(0, 0)),
            2 => Some(self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0)),
            3 => {
                let [[a, b, c], [d, e, f], [g, h, i]] = self.to_array3x3()?;
                Some(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))
            }
            _ => self.determinant_laplace(),
        }
    }

    /// 使用Laplace展开计算行列式
    fn determinant_laplace(&self) -> Option<f64> {
        if self.rows != self.cols {
            return None;
        }

        let n = self.rows;
        if n == 1 {
            return Some(self.get(0, 0));
        }

        let mut det = 0.0;
        for j in 0..n {
            let minor = self.minor(0, j);
            let cofactor = if j % 2 == 0 { 1.0 } else { -1.0 } * self.get(0, j);
            det += cofactor * minor.determinant().unwrap_or(0.0);
        }
        Some(det)
    }

    /// 计算余子式
    /// 
    fn minor(&self, row: usize, col: usize) -> MatMN {
        let mut result = MatMN::new(self.rows - 1, self.cols - 1);
        let mut r = 0;
        for i in 0..self.rows {
            if i == row {
                continue;
            }
            let mut c = 0;
            for j in 0..self.cols {
                if j == col {
                    continue;
                }
                result.set(r, c, self.get(i, j));
                c += 1;
            }
            r += 1;
        }
        result
    }

    /// 转换为3x3数组（如果可能）
    fn to_array3x3(&self) -> Option<[[f64; 3]; 3]> {
        if self.rows != 3 || self.cols != 3 {
            return None;
        }

        Some([
            [self.get(0, 0), self.get(0, 1), self.get(0, 2)],
            [self.get(1, 0), self.get(1, 1), self.get(1, 2)],
            [self.get(2, 0), self.get(2, 1), self.get(2, 2)],
        ])
    }

    /// 计算矩阵的逆（仅适用于方阵）
    pub fn inverse(&self) -> Option<MatMN> {
        if self.rows != self.cols {
            return None;
        }

        let det = self.determinant()?;
        if det.abs() < 1e-10 {
            return None;
        }

        let n = self.rows;
        let mut result = MatMN::new(n, n);

        for i in 0..n {
            for j in 0..n {
                let minor = self.minor(i, j);
                let cofactor = if (i + j) % 2 == 0 { 1.0 } else { -1.0 } * minor.determinant()?;
                result.set(j, i, cofactor / det); // 注意这里交换了i,j以同时完成转置
            }
        }

        Some(result)
    }

    /// 计算矩阵的秩
    pub fn rank(&self) -> usize {
        let mut mat = self.clone();
        let mut rank = 0;
        let mut row = 0;
        let mut col = 0;

        while row < self.rows && col < self.cols {
            // 找到当前列中绝对值最大的元素
            let mut max_row = row;
            let mut max_val = mat.get(row, col).abs();

            for i in (row + 1)..self.rows {
                let val = mat.get(i, col).abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            if max_val < 1e-10 {
                col += 1;
                continue;
            }

            // 交换行
            if max_row != row {
                for j in 0..self.cols {
                    let temp = mat.get(row, j);
                    mat.set(row, j, mat.get(max_row, j));
                    mat.set(max_row, j, temp);
                }
            }

            // 消元
            for i in (row + 1)..self.rows {
                let factor = mat.get(i, col) / mat.get(row, col);
                for j in col..self.cols {
                    let val = mat.get(i, j) - factor * mat.get(row, j);
                    mat.set(i, j, val);
                }
            }

            rank += 1;
            row += 1;
            col += 1;
        }

        rank
    }

    /// LU分解
    /// 返回 (L, U)，其中L是下三角矩阵，U是上三角矩阵
    pub fn lu_decomposition(&self) -> Option<(MatMN, MatMN)> {
        if self.rows != self.cols {
            return None;
        }

        let n = self.rows;
        let mut l = MatMN::new(n, n);
        let mut u = self.clone();

        for i in 0..n {
            // L矩阵对角线元素设为1
            l.set(i, i, 1.0);

            for j in i..n {
                let mut sum = 0.0;
                for k in 0..i {
                    sum += l.get(i, k) * u.get(k, j);
                }
                u.set(i, j, u.get(i, j) - sum);
            }

            for j in (i + 1)..n {
                let mut sum = 0.0;
                for k in 0..i {
                    sum += l.get(j, k) * u.get(k, i);
                }
                if u.get(i, i).abs() < 1e-10 {
                    return None;
                }
                l.set(j, i, (self.get(j, i) - sum) / u.get(i, i));
            }
        }

        Some((l, u))
    }

    /// QR分解
    /// 返回 (Q, R)，其中Q是正交矩阵，R是上三角矩阵
    pub fn qr_decomposition(&self) -> Option<(MatMN, MatMN)> {
        if self.rows < self.cols {
            return None;
        }

        let m = self.rows;
        let n = self.cols;
        let mut q = MatMN::new(m, n);
        let mut r = MatMN::new(n, n);

        // Gram-Schmidt正交化
        for j in 0..n {
            let mut v = self.get_col(j);
            
            for i in 0..j {
                let qi = q.get_col(i);
                let r_ij = dot_product(&qi, &self.get_col(j));
                r.set(i, j, r_ij);
                
                for k in 0..m {
                    v[k] -= r_ij * qi[k];
                }
            }

            let norm = euclidean_norm(&v);
            if norm < 1e-10 {
                return None;
            }

            r.set(j, j, norm);
            for i in 0..m {
                q.set(i, j, v[i] / norm);
            }
        }

        Some((q, r))
    }

    /// 特征值计算（使用QR算法）
    pub fn eigenvalues(&self, max_iterations: usize) -> Option<Vec<f64>> {
        if self.rows != self.cols {
            return None;
        }

        let mut a = self.clone();
        let n = self.rows;
        let tolerance = 1e-10;

        for _ in 0..max_iterations {
            if let Some((q, r)) = a.qr_decomposition() {
                let next_a = if let Some(prod) = &r * &q {
                    prod
                } else {
                    return None;
                };

                // 检查对角线外元素是否足够小
                let mut converged = true;
                for i in 0..n {
                    for j in 0..n {
                        if i != j && next_a.get(i, j).abs() > tolerance {
                            converged = false;
                            break;
                        }
                    }
                    if !converged {
                        break;
                    }
                }

                if converged {
                    let mut eigenvalues = Vec::with_capacity(n);
                    for i in 0..n {
                        eigenvalues.push(next_a.get(i, i));
                    }
                    return Some(eigenvalues);
                }

                a = next_a;
            } else {
                return None;
            }
        }

        None
    }

    /// 奇异值分解 (SVD)
    /// 返回 (U, Σ, V^T)
    pub fn svd(&self, max_iterations: usize) -> Option<(MatMN, Vec<f64>, MatMN)> {
        let m = self.rows;
        let n = self.cols;

        // 计算A^T * A的特征值和特征向量
        let ata = if let Some(prod) = &self.transpose() * self {
            prod
        } else {
            return None;
        };

        // 计算特征值
        let eigenvalues = ata.eigenvalues(max_iterations)?;
        let mut singular_values: Vec<f64> = eigenvalues.iter()
            .map(|&x| x.abs().sqrt())
            .collect();
        singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // 构造Σ矩阵（仅返回奇异值）
        
        // 计算V（特征向量）
        let mut v = MatMN::new(n, n);
        // TODO: 实现特征向量的计算
        
        // 计算U
        let mut u = MatMN::new(m, m);
        for i in 0..std::cmp::min(m, n) {
            if singular_values[i] > 1e-10 {
                let v_col = v.get_col(i);
                let col = if let Some(prod) = self * &v_col {
                    prod
                } else {
                    continue;
                };
                let norm = euclidean_norm(&col);
                for j in 0..m {
                    u.set(j, i, col[j] / norm);
                }
            }
        }

        Some((u, singular_values, v.transpose()))
    }

    /// 求解线性方程组 Ax = b
    pub fn solve(&self, b: &Vec<f64>) -> Option<Vec<f64>> {
        if self.rows != self.cols || self.rows != b.len() {
            return None;
        }

        // 使用LU分解求解
        let (l, u) = self.lu_decomposition()?;
        
        // 前向替换求解Ly = b
        let mut y = vec![0.0; self.rows];
        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l.get(i, j) * y[j];
            }
            y[i] = b[i] - sum;
        }

        // 后向替换求解Ux = y
        let mut x = vec![0.0; self.rows];
        for i in (0..self.rows).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..self.rows {
                sum += u.get(i, j) * x[j];
            }
            if u.get(i, i).abs() < 1e-10 {
                return None;
            }
            x[i] = (y[i] - sum) / u.get(i, i);
        }

        Some(x)
    }

    /// 计算矩阵的条件数
    pub fn condition_number(&self) -> Option<f64> {
        if self.rows != self.cols {
            return None;
        }

        // 使用奇异值计算条件数
        let (_, singular_values, _) = self.svd(100)?;
        let max_sv = singular_values[0];
        let min_sv = *singular_values.last()?;

        if min_sv < 1e-10 {
            return None;
        }

        Some(max_sv / min_sv)
    }
}

// 辅助函数：计算向量的点积
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// 辅助函数：计算向量的欧几里得范数
fn euclidean_norm(v: &[f64]) -> f64 {
    dot_product(v, v).sqrt()
}

// 矩阵加法
impl Add for &MatMN {
    type Output = Option<MatMN>;

    fn add(self, rhs: &MatMN) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return None;
        }

        let mut result = MatMN::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) + rhs.get(i, j));
            }
        }
        Some(result)
    }
}

// 矩阵减法
impl Sub for &MatMN {
    type Output = Option<MatMN>;

    fn sub(self, rhs: &MatMN) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return None;
        }

        let mut result = MatMN::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) - rhs.get(i, j));
            }
        }
        Some(result)
    }
}

// 矩阵乘法
impl Mul for &MatMN {
    type Output = Option<MatMN>;

    fn mul(self, rhs: &MatMN) -> Self::Output {
        if self.cols != rhs.rows {
            return None;
        }

        let mut result = MatMN::new(self.rows, rhs.cols);
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * rhs.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        Some(result)
    }
}

// 矩阵与标量相乘
impl Mul<f64> for &MatMN {
    type Output = MatMN;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut result = MatMN::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) * rhs);
            }
        }
        result
    }
}

// 矩阵与向量相乘
impl Mul<&Vec<f64>> for &MatMN {
    type Output = Option<Vec<f64>>;

    fn mul(self, rhs: &Vec<f64>) -> Self::Output {
        if self.cols != rhs.len() {
            return None;
        }

        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.get(i, j) * rhs[j];
            }
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let data = &[
            &[1.0_f64, 2.0, 3.0][..],
            &[4.0, 5.0, 6.0][..],
        ];
        let mat = MatMN::from_array(data);
        assert_eq!(mat.size(), (2, 3));
        assert_eq!(mat.get(0, 0), 1.0);
        assert_eq!(mat.get(1, 2), 6.0);
    }

    #[test]
    fn test_matrix_operations() {
        let mat1 = MatMN::from_array(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let mat2 = MatMN::from_array(&[&[5.0, 6.0], &[7.0, 8.0]]);

        // 加法
        let sum = (&mat1 + &mat2).unwrap();
        assert_eq!(sum.get(0, 0), 6.0);
        assert_eq!(sum.get(1, 1), 12.0);

        // 乘法
        let prod = (&mat1 * &mat2).unwrap();
        assert_eq!(prod.get(0, 0), 19.0);
        assert_eq!(prod.get(0, 1), 22.0);
        assert_eq!(prod.get(1, 0), 43.0);
        assert_eq!(prod.get(1, 1), 50.0);
    }

    #[test]
    fn test_matrix_determinant() {
        let mat = MatMN::from_array(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
        ]);
        assert!((mat.determinant().unwrap()).abs() < 1e-10);

        let mat = MatMN::from_array(&[
            &[2.0, -1.0],
            &[-1.0, 3.0],
        ]);
        assert_eq!(mat.determinant().unwrap(), 5.0);
    }

    #[test]
    fn test_matrix_rank() {
        let mat = MatMN::from_array(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
        ]);
        assert_eq!(mat.rank(), 2); // 因为第三行是前两行的线性组合

        let mat = MatMN::from_array(&[
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
        ]);
        assert_eq!(mat.rank(), 3); // 满秩矩阵
    }

    #[test]
    fn test_matrix_inverse() {
        let mat = MatMN::from_array(&[
            &[4.0, 7.0],
            &[2.0, 6.0],
        ]);
        let inv = mat.inverse().unwrap();
        let prod = (&mat * &inv).unwrap();
        
        // 验证 M * M^(-1) = I
        assert!((prod.get(0, 0) - 1.0).abs() < 1e-10);
        assert!(prod.get(0, 1).abs() < 1e-10);
        assert!(prod.get(1, 0).abs() < 1e-10);
        assert!((prod.get(1, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lu_decomposition() {
        let mat = MatMN::from_array(&[
            &[4.0, 3.0],
            &[6.0, 3.0],
        ]);

        let (l, u) = mat.lu_decomposition().unwrap();
        let prod = (&l * &u).unwrap();

        // 验证 L * U = A
        for i in 0..2 {
            for j in 0..2 {
                assert!((prod.get(i, j) - mat.get(i, j)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_qr_decomposition() {
        let mat = MatMN::from_array(&[
            &[12.0, -51.0, 4.0],
            &[6.0, 167.0, -68.0],
            &[-4.0, 24.0, -41.0],
        ]);

        let (q, r) = mat.qr_decomposition().unwrap();
        let prod = (&q * &r).unwrap();

        // 验证 Q * R = A
        for i in 0..3 {
            for j in 0..3 {
                assert!((prod.get(i, j) - mat.get(i, j)).abs() < 1e-10);
            }
        }

        // 验证Q是正交矩阵
        let qt = q.transpose();
        let q_qt = (&q * &qt).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((q_qt.get(i, j) - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_solve_linear_system() {
        let mat = MatMN::from_array(&[
            &[3.0, 2.0],
            &[1.0, -1.0],
        ]);
        let b = vec![7.0, 1.0];

        let x = mat.solve(&b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }
} 
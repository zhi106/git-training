pub mod vec_mathz;
use vec_mathz::{Vec3, Mat3, MatMN};

fn main() {
    // 基本构造
    println!("=== 基本构造 ===");
    let v1 = Vec3::new(1.0, 2.0, 3.0);
    let v2 = Vec3::new(2.0, 4.0, 6.0);
    println!("v1: {:?}", v1);
    println!("v2: {:?}", v2);

    // 基本运算
    println!("\n=== 基本运算 ===");
    let sum = v1 + v2;
    println!("v1 + v2 = {:?}", sum);

    let diff = v1 - v2;
    println!("v1 - v2 = {:?}", diff);

    let neg = -v1;
    println!("-v1 = {:?}", neg);

    // 标量运算
    println!("\n=== 标量运算 ===");
    let scaled = v1 * 2.0;
    println!("v1 * 2.0 = {:?}", scaled);

    let divided = v1 / 2.0;
    println!("v1 / 2.0 = {:?}", divided);

    // 向量运算
    println!("\n=== 向量运算 ===");
    let dot_product = v1 * v2;
    println!("v1 · v2 = {}", dot_product);

    let cross_product = v1 ^ v2;
    println!("v1 × v2 = {:?}", cross_product);

    // 几何运算
    println!("\n=== 几何运算 ===");
    println!("v1的长度: {}", v1.length());
    println!("v1的单位向量: {:?}", v1.normalize());

    let angle = v1.angle(&v2);
    println!("v1和v2的夹角（弧度）: {}", angle);
    println!("v1和v2的夹角（角度）: {}", angle.to_degrees());

    // 向量变换
    println!("\n=== 向量变换 ===");
    let normal = Vec3::unit_y();
    let reflected = v1.reflect(&normal);
    println!("v1关于y轴的反射: {:?}", reflected);

    let projected = v1.project_onto(&v2);
    println!("v1在v2上的投影: {:?}", projected);

    // 向量关系
    println!("\n=== 向量关系 ===");
    println!("v1和v2是否平行: {}", v1.is_parallel(&v2));
    println!("v1和v2是否垂直: {}", v1.is_perpendicular(&v2));
    println!("v1到v2的距离: {}", v1.distance(&v2));

    // 赋值运算
    println!("\n=== 赋值运算 ===");
    let mut v3 = v1;
    v3 += v2;
    println!("v3 += v2: {:?}", v3);

    v3 *= 2.0;
    println!("v3 *= 2.0: {:?}", v3);

    // 数组转换
    println!("\n=== 数组转换 ===");
    let arr = [1.0, 2.0, 3.0];
    let v4 = Vec3::from_array(arr);
    println!("从数组创建向量: {:?}", v4);
    println!("向量转换为数组: {:?}", v4.to_array());

    // 矩阵操作示例
    println!("\n=== 矩阵操作 ===");
    
    // 创建矩阵
    let m1 = Mat3::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]);
    println!("矩阵 m1:\n{:?}", m1);

    // 单位矩阵
    let identity = Mat3::identity();
    println!("\n单位矩阵:\n{:?}", identity);

    // 矩阵乘法
    let m2 = Mat3::new([
        [9.0, 8.0, 7.0],
        [6.0, 5.0, 4.0],
        [3.0, 2.0, 1.0],
    ]);
    let m3 = m1 * m2;
    println!("\n矩阵乘法 m1 * m2:\n{:?}", m3);

    // 矩阵转置
    let m1_t = m1.transpose();
    println!("\n矩阵m1的转置:\n{:?}", m1_t);

    // 矩阵行列式
    let det = m1.determinant();
    println!("\n矩阵m1的行列式: {}", det);

    // 矩阵求逆
    if let Some(inv) = m1.inverse() {
        println!("\n矩阵m1的逆:\n{:?}", inv);
    } else {
        println!("\n矩阵m1不可逆");
    }

    // 矩阵与向量乘法
    let v = Vec3::new(1.0, 2.0, 3.0);
    let transformed = m1 * v;
    println!("\n矩阵与向量乘法 m1 * v:\n{:?}", transformed);

    // 旋转矩阵示例
    let angle = std::f64::consts::PI / 4.0; // 45度
    let rot_x = Mat3::rotation_x(angle);
    let rot_y = Mat3::rotation_y(angle);
    let rot_z = Mat3::rotation_z(angle);

    println!("\n绕X轴旋转45度的矩阵:\n{:?}", rot_x);
    println!("\n绕Y轴旋转45度的矩阵:\n{:?}", rot_y);
    println!("\n绕Z轴旋转45度的矩阵:\n{:?}", rot_z);

    // 缩���矩阵示例
    let scale = Mat3::scaling(2.0, 3.0, 4.0);
    println!("\n缩放矩阵 (sx=2, sy=3, sz=4):\n{:?}", scale);

    // 复合变换示例
    let v = Vec3::new(1.0, 0.0, 0.0);
    let transformed = rot_z * scale * v;
    println!("\n向量经过缩放和旋转后:\n{:?}", transformed);

    // 通用矩阵示例
    println!("\n=== 通用矩阵操作 ===");
    
    // 创建非方阵
    let data1 = &[
        &[1.0, 2.0, 3.0][..],
        &[4.0, 5.0, 6.0][..],
    ];
    let m1 = MatMN::from_array(data1);
    println!("矩阵 m1 (2x3):\n{:?}", m1);

    let data2 = &[
        &[7.0, 8.0][..],
        &[9.0, 10.0][..],
        &[11.0, 12.0][..],
    ];
    let m2 = MatMN::from_array(data2);
    println!("\n矩阵 m2 (3x2):\n{:?}", m2);

    // 矩阵乘法
    if let Some(prod) = &m1 * &m2 {
        println!("\n矩阵乘法 m1 * m2 (2x2):\n{:?}", prod);
    }

    // 矩阵转置
    let m1_t = m1.transpose();
    println!("\n矩阵m1的转置 (3x2):\n{:?}", m1_t);

    // 矩阵秩
    println!("\n矩阵m1的秩: {}", m1.rank());

    // 方阵示例
    let square_data = &[
        &[1.0, 2.0, 3.0][..],
        &[0.0, 4.0, 5.0][..],
        &[0.0, 0.0, 6.0][..],
    ];
    let square = MatMN::from_array(square_data);
    println!("\n上三角矩阵:\n{:?}", square);

    // 行列式
    if let Some(det) = square.determinant() {
        println!("\n行列式: {}", det);
    }

    // 矩阵求逆
    if let Some(inv) = square.inverse() {
        println!("\n矩阵的逆:\n{:?}", inv);
        
        // 验证求逆结果
        if let Some(prod) = &square * &inv {
            println!("\n验证 A * A^(-1) = I:\n{:?}", prod);
        }
    }

    // 线性相关性示例
    let dependent_data = &[
        &[1.0, 2.0, 3.0][..],
        &[2.0, 4.0, 6.0][..],  // 第二行是第一行的2倍
        &[3.0, 6.0, 9.0][..],  // 第三行是第一行的3倍
    ];
    let dependent = MatMN::from_array(dependent_data);
    println!("\n线性相关矩阵:\n{:?}", dependent);
    println!("矩阵的秩: {}", dependent.rank());  // 应该是1

    // 矩阵分解示例
    println!("\n=== 矩阵分解 ===");
    
    // LU分解示例
    let mat = MatMN::from_array(&[
        &[4.0, 3.0][..],
        &[6.0, 3.0][..],
    ]);
    println!("\n原始矩阵:\n{:?}", mat);

    if let Some((l, u)) = mat.lu_decomposition() {
        println!("\nLU分解结果:");
        println!("L矩阵:\n{:?}", l);
        println!("U矩阵:\n{:?}", u);
        
        // 验证 L * U = A
        if let Some(prod) = &l * &u {
            println!("验证 L * U:\n{:?}", prod);
        }
    }

    // QR分解示例
    let mat = MatMN::from_array(&[
        &[12.0, -51.0,   4.0][..],
        &[ 6.0, 167.0, -68.0][..],
        &[-4.0,  24.0, -41.0][..],
    ]);
    println!("\nQR分解原始矩阵:\n{:?}", mat);

    if let Some((q, r)) = mat.qr_decomposition() {
        println!("\nQR分解结果:");
        println!("Q矩阵（正交矩阵）:\n{:?}", q);
        println!("R矩阵（上三角）:\n{:?}", r);
        
        // 验证 Q * R = A
        if let Some(prod) = &q * &r {
            println!("验证 Q * R:\n{:?}", prod);
        }
    }

    // 特征值计算示例
    let mat = MatMN::from_array(&[
        &[2.0, -1.0][..],
        &[-1.0, 2.0][..],
    ]);
    println!("\n计算特征值的矩阵:\n{:?}", mat);

    if let Some(eigenvalues) = mat.eigenvalues(100) {
        println!("特征值: {:?}", eigenvalues);
    }

    // 线性方程组求解示例
    let mat = MatMN::from_array(&[
        &[3.0, 2.0][..],
        &[1.0, -1.0][..],
    ]);
    let b = vec![7.0, 1.0];
    println!("\n求解线性方程组 Ax = b");
    println!("A:\n{:?}", mat);
    println!("b: {:?}", b);

    if let Some(x) = mat.solve(&b) {
        println!("解 x: {:?}", x);
        
        // 验证解
        if let Some(prod) = &mat * &MatMN::from_array(&[&x[..]]) {
            println!("验证 Ax:\n{:?}", prod);
        }
    }

    // 矩阵条件数计算示例
    let mat = MatMN::from_array(&[
        &[4.0, 3.0][..],
        &[3.0, 2.0][..],
    ]);
    println!("\n计算条件数的矩阵:\n{:?}", mat);

    if let Some(cond) = mat.condition_number() {
        println!("条件数: {}", cond);
    }
}

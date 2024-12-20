//! 向量运算符实现模块
//! 
//! 实现了向量的各种运算符重载

use super::Vec3;
use std::ops::*;

/// 实现所有运算符特征的宏
macro_rules! impl_op_traits {
    ($t:ty) => {
        /// 实现向量取负
        impl Neg for $t {
            type Output = $t;
            #[inline]
            fn neg(self) -> Self::Output {
                Vec3::new(-self.x, -self.y, -self.z)
            }
        }

        /// 实现向量加法
        impl Add for $t {
            type Output = $t;
            #[inline]
            fn add(self, other: Self) -> Self::Output {
                Vec3::new(
                    self.x + other.x,
                    self.y + other.y,
                    self.z + other.z,
                )
            }
        }

        /// 实现向量减法
        impl Sub for $t {
            type Output = $t;
            #[inline]
            fn sub(self, other: Self) -> Self::Output {
                Vec3::new(
                    self.x - other.x,
                    self.y - other.y,
                    self.z - other.z,
                )
            }
        }

        /// 实现向量与标量的乘法
        impl Mul<f64> for $t {
            type Output = $t;
            #[inline]
            fn mul(self, scalar: f64) -> Self::Output {
                Vec3::new(
                    self.x * scalar,
                    self.y * scalar,
                    self.z * scalar,
                )
            }
        }

        /// 实现标量与向量的乘法
        impl Mul<$t> for f64 {
            type Output = $t;
            #[inline]
            fn mul(self, vec: $t) -> Self::Output {
                vec * self
            }
        }

        /// 实现向量点乘
        impl Mul for $t {
            type Output = f64;
            #[inline]
            fn mul(self, other: Self) -> Self::Output {
                self.dot(&other)
            }
        }

        /// 实现向量与标量的除法
        impl Div<f64> for $t {
            type Output = $t;
            #[inline]
            fn div(self, scalar: f64) -> Self::Output {
                Vec3::new(
                    self.x / scalar,
                    self.y / scalar,
                    self.z / scalar,
                )
            }
        }

        /// 实现向量叉乘
        impl BitXor for $t {
            type Output = $t;
            #[inline]
            fn bitxor(self, other: Self) -> Self::Output {
                self.cross(&other)
            }
        }

        /// 实现向量加法赋值
        impl AddAssign for $t {
            #[inline]
            fn add_assign(&mut self, other: Self) {
                self.x += other.x;
                self.y += other.y;
                self.z += other.z;
            }
        }

        /// 实现向量减法赋值
        impl SubAssign for $t {
            #[inline]
            fn sub_assign(&mut self, other: Self) {
                self.x -= other.x;
                self.y -= other.y;
                self.z -= other.z;
            }
        }

        /// 实现向量与标量的乘法赋值
        impl MulAssign<f64> for $t {
            #[inline]
            fn mul_assign(&mut self, scalar: f64) {
                self.x *= scalar;
                self.y *= scalar;
                self.z *= scalar;
            }
        }

        /// 实现向量与标量的除法赋值
        impl DivAssign<f64> for $t {
            #[inline]
            fn div_assign(&mut self, scalar: f64) {
                self.x /= scalar;
                self.y /= scalar;
                self.z /= scalar;
            }
        }
    };
}

// 实现所有运算符特征
impl_op_traits!(Vec3); 
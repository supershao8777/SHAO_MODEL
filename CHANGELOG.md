# CHANGELOG

## v1.0 (2026-07) — RCAGL 基础
- **RCAGL**: 鲁棒一致性锚点图学习 (TKDE 2024)
  - 交替优化 P/A/C/D，核范数正则 + 跨视图 L1
  - 二部图谱共聚类 C 更新
  - 8 种聚类评估指标

## v1.1 (2026-07) — MVC-AO
- **新增 MVC-AO**: 基于交替优化的多视图聚类
  - 共享聚类原型 Ps + 视图偏移 Q + 锚点字典 A
  - CG 求解器避免 Kronecker 积
  - R 为软聚类指示矩阵 (概率单纯形)
- **RCAGL 优化**: 收敛诊断输出 + P 微量 ridge + C 单纯形安全投影

## v1.2 (2026-07) — BCD-MVRL v1
- **新增 BCD-MVRL v1**: 正交多视图表示学习
  - 正交 R(n×r) + 正交 Ps(r×m) + 非负 Q/A
  - L1 跨视图冗余惩罚 α Σ ||Q^(v) ⊙ Q^(u)||₁
  - Q 近端梯度 + Soft⁺ 非负投影
- **MVC-AO**: Ps 移除正交约束 → 闭式双边除法 Ps = M\G/S

## v1.3 (2026-07) — BCD-MVRL v2/v3/v4
- **BCD-MVRL v2**: 视图特有正交投影 R^(v) + 共识 S + Q^T Ps=0 零空间约束
- **BCD-MVRL v3**: 移除 Q 零空间约束，Ps/Q 改用闭式双边除法 (无梯度下降)
- **BCD-MVRL v4**: 恢复 R^T R=I_m 正交约束，全部更新闭式解 (5 步 BCD)

## v1.4 (2026-07) — BCD-MVRL v5~v11 系列

| 版本 | 核心创新 | Ps 更新 | Q 更新 | A 更新 | 特殊约束 |
|------|----------|---------|--------|--------|----------|
| **v5** | 无正交 + β 正则 | GD | GD | 闭式+simplex | β\|Ps^T Q\|² |
| **v6** | Kronecker Sylvester Q | GD+SVD | **Sylvester 精确解** | 闭式+simplex | Stiefel Ps |
| **v7** | L21 行稀疏 Q | GD+SVD | 近端梯度(组软阈值) | 闭式+simplex | L21\|Q\|, Ps-only 共识 |
| **v8** | IRLS L21(QA) | GD+SVD | GD(IRLS 权重) | IRLS 闭式 | L21\|QA\|, 正交 R |
| **v9** | 视图权重 r_v | GD+SVD | Kronecker Sylvester | PGD | r_v 逆误差, λ₃\|R\|² |
| **v10** | Cayley + NMF Q | **Cayley** | NMF 乘法(pos/neg) | PGD | Q≥0, Ps Stiefel |
| **v11** | Riemannian + Duchi | **Riemannian+QR** | 闭式+ReLU | PGD+Duchi | Duchi 单纯形, Procrustes R |

### v1.4 优化改进
- **bcd_mvrl3 重构**: 单调下降 BCD (回溯线搜索/阻尼保护/自适应 ridge)
- 全部 demo 脚本支持网格搜索 + 收敛诊断
- Frobenius 范数数据归一化

### 算法演化关系
```
v1.0 RCAGL (锚点图 + 交替最小化)
  ↓
v1.1 MVC-AO (共享原型 + CG求解)
  ↓
v1.2 BCD-MVRL v1 (正交表示 + L1冗余)
  ↓
v1.3 BCD-MVRL v2→v4 (视图特有投影 + 共识S + 闭式解)
  ↓
v1.4 BCD-MVRL v5→v11 (IRLS/L21/Sylvester/Cayley/Riemannian)
```

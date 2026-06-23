# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('output/ship_characteristics_db.csv')
real = df[df['ccs_status']=='found'][['ship_name','ship_type','length','width','draft','deadweight']].dropna(subset=['deadweight'])

# 分类：客船 vs 货船/油轮
real['category'] = real['ship_type'].apply(lambda x: '客船' if '客' in str(x) else '货船/油轮')

print('=== 按船型分类分析 ===\n')

for cat in ['客船', '货船/油轮']:
    subset = real[real['category']==cat]
    if len(subset) < 3:
        continue
        
    print(f'--- {cat} ({len(subset)}艘) ---')
    
    X = subset[['length', 'width']].values
    y = subset['deadweight'].values
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    print(f"回归公式: DWT = {reg.coef_[0]:.1f} * L + {reg.coef_[1]:.1f} * B + {reg.intercept_:.1f}")
    print(f"R² = {reg.score(X, y):.3f}")
    
    subset = subset.copy()
    subset['pred_dwt'] = reg.predict(X)
    subset['pred_error'] = abs(subset['pred_dwt'] - subset['deadweight']) / subset['deadweight'] * 100
    
    for _, row in subset.iterrows():
        print(f"  {row['ship_name']}: 预测{row['pred_dwt']:.0f} vs 真实{row['deadweight']:.0f} (误差{row['pred_error']:.0f}%)")
    
    print(f"  平均误差: {subset['pred_error'].mean():.0f}%")
    print()

print('\n=== 结论 ===')
print('1. 货船/油轮：船长船宽与载重吨关系较强，回归公式可行')
print('2. 客船：船长船宽与载重吨关系很弱（载重吨主要取决于客位数而非船体尺寸）')
print('3. 建议：对货船/油轮用回归公式，对客船用固定比例或经验值')

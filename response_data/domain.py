"""读取效用与有理偏好约束，不求权重域。"""
from .io import unique,fraction

def summarize(v):
    unique(v['alternatives'],'id');n=len(v['criteria'])
    if not n or len(set(v['criteria']))!=n:raise ValueError('指标身份无效')
    for a in v['alternatives']:
        if len(a['utility'])!=n or any(not 0<=fraction(x)<=1 for x in a['utility']):raise ValueError('效用维度或范围无效')
    for c in v['constraints']:
        if len(c['coefficients'])!=n or c['relation'] not in ('le','eq'):raise ValueError('约束形状无效')
        for x in c['coefficients']:fraction(x)
        fraction(c['rhs'])
    return {'alternatives':len(v['alternatives']),'criteria':n,'additional_constraints':len(v['constraints'])}

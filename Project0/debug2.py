# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 20:13:30 2019

@author: Ender
"""

def get_sum_metrics(predictions, metrics=()):
#    metrics=[]
    for i in range(3):
        def f(x,i=i): return x+i
        metrics +=(f,)

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)
    return sum_metrics
    
print(get_sum_metrics(1))  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
print(get_sum_metrics(2))  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9
print(get_sum_metrics(3, [lambda x: x]))  # Should be (3) + (3 + 0) + (3 + 1) + (3 + 2) = 15
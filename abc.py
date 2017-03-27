#!/bin/python

import sys
import os

def  countSum(numbers):
    sumall=0
    for num in numbers:
        div=[]
        for i in range(1,num,2):
            # print i
            if num%i==0:
                div.append(i)
                # print num
        if num%2!=0:
            div.append(num)
            print div
        sumall+=sum(div)
        print sumall
    print sumall

countSum([1,3,7])
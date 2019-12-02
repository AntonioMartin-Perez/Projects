# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 11:23:33 2017

@author: yomis
"""
def McNuggets(n):
    """
    n is an int

    Returns True if some integer combination of 6, 9 and 20 equals n
    Otherwise returns False.
    """
    a=0
    b=0
    c=0
    rem=n
    
    if n%6==0 or n%9==0 or n%20==0:
        return True
    if n%2==1 and n>9:
        b+=1
        
    while True:
#        print(a,b,c)
        rem =-(6*a+9*b+20*c-n)
#        print(rem)
        if rem==0:
            return True
        elif rem%2==1 and rem>9:
            b+=1
        elif rem>=20:
            c+=1
        elif rem>=6 and rem<20:
            a+=1
        else:
            break
    for x in range (0,max(a,b,c)):
        for y in range(0,max(a,b,c)):
            for z in range(0,max(a,b,c)):
                if 6*x+9*y+20*z==n:
                    return True
    return False

    
def longestRun(L):
    longest=[L[0]]
    test=[L[0]]
    for i in range(1,len(L)):
        if L[i]>=test[-1]:
            test.append(L[i])
            if len(test)>=len(longest):
                longest=test
        else:
            test=[L[i]]
            
    return len(longest)


def cipher(map_from, map_to, code):
    """ map_from, map_to: strings where each contain 
                          N unique lowercase letters. 
        code: string (assume it only contains letters also in map_from)
        Returns a tuple of (key_code, decoded).
        key_code is a dictionary with N keys mapping str to str where 
        each key is a letter in map_from at index i and the corresponding 
        value is the letter in map_to at index i. 
        decoded is a string that contains the decoded version 
        of code using the key_code mapping. """
    dict={}
    codigo=''
    i=0
    for char in map_from:
        dict[char]=map_to[i]
        i+=1
    for letter in code:
        codigo+=dict[letter]
        
    tupla=(dict,codigo)
    
        
    return tupla


class Person(object):     
    def __init__(self, name):         
        self.name = name     
    def say(self, stuff):         
        return self.name + ' says: ' + stuff     
    def __str__(self):         
        return self.name  

class Lecturer(Person):     
    def lecture(self, stuff):         
        return 'I believe that ' + Person.say(self, stuff)  

class Professor(Lecturer): 
    def say(self, stuff): 
        return self.name + ' says: ' + self.lecture(stuff)


class ArrogantProfessor(Professor): 
    def say(self, stuff): 
        return self.name + ' says: ' + self.watever(stuff)
    def lecture(self, stuff): 
        return 'It is obvious that ' + Person.say(self,stuff)
    def watever(self,stuff):
        return self.lecture(stuff)

    
        
    
        
        
        
          
        
    
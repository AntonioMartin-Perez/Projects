# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 10:47:01 2016

@author: pffff
"""

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
        return 'Prof. ' + self.name + ' says: ' + self.lecture(stuff)

class ArrogantProfessor(Professor): 
    def say(self, stuff): 
        return self.name + ' says: ' + self.lecture(stuff)
    def lecture(self, stuff): 
        return 'It is obvious that ' + Lecturer.lecture(self,stuff)
        

pe = Professor('eric') 
print(pe.say('the sky is blue'))
    

ae = ArrogantProfessor('eric')
print(ae.say('the sky is blue'))
print(ae.lecture('the sky is blue'))

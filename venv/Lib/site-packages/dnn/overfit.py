import sys
import numpy as np

class Overfit:
    def __init__ (self, keep_count = 0, period = 5, min_validations = 0):      
        self.keep_count = keep_count
        self.period = period
        self.min_validations = min_validations
        
        self.validations = 0
        self.overfitted_count = 0
        self.cost_log = [[], []]        
        self.new_record = False
        self.min_cost = sys.maxsize
        self.overfit = False
    
    @property
    def latest_cost (self):
        return  self.cost_log [1][-1]
        
    def is_overfit (self):
        return self.overfit
    
    def is_renewaled (self):
        return self.new_record
    
    def add_cost (self, cost, is_validating):
        if not is_validating: # currently recall loss is not used
            return
        
        self.validations += 1       
        self.overfit, self.new_record = False, False        
        if is_validating:
            if cost < self.min_cost: 
                self.min_cost = cost   
                self.new_record = True
                self.overfitted_count = 0 # reset count
        
        index = int (is_validating)
        if self.cost_log [index]:
            latest = np.mean (self.cost_log [index])
        self.cost_log [index].append (cost)
        if len (self.cost_log  [index]) < self.period:
            return
        
        self.cost_log [index] = self.cost_log [index] [-self.period:]        
        if self.min_validations and self.validations < self.min_validations:
            return
        
        current = np.mean (self.cost_log [index])
        if current >= latest:
            self.overfitted_count += 1            
            if self.keep_count and self.overfitted_count > self.keep_count:
                self.overfit = True        
        else:
            self.overfitted_count = 0
        
        #print (self.cost_log)
        #print (latest, current)
        #print (self.overfitted_count, self.overfit)
        
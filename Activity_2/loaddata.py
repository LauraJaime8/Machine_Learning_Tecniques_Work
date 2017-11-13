# -*- coding: utf-8 -*-

"""

@author: Ruth Rodríguez-Manzaneque López, Diego Andérica Richard y Laura Jaime Villamayor

"""
import codecs

def load_data(path):
        
    f = codecs.open(path, "r", "utf-8")
    records = []
    names = []
    years = ["1997", "1998", "1999", "2000", "2001", "2002", "2003"]
    first_row = True
    
    for line in f:
        
        if first_row == True:
            row = line.split(",")
            
            for i in range(4, len(row)):
                names.append(row[i])
                
            first_row = False
        else:
            #Replace no-data fields
            while ",," in line:
                line = line.replace(",,", ",0,")
        
            #Replace last unfilled field
            line = line.replace(",\n", ",0\n")
           
            row = line.split(",")
            
            if row[0] == "sj" and row[1] in years:
                for i in range (4):
                    row.pop(0)
            
                if row != []:
                    records.append(map(float, row))

    return records, names

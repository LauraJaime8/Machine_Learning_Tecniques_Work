# -*- coding: utf-8 -*-

"""

@author Diego Andérica Richard, Ruth Rodríguez-Manzaneque López, Laura Jaime Villamayor

"""

import codecs

def load_data(path):
        
    f = codecs.open(path, "r", "utf-8")
    records = []
    names = []
    years = ["1997", "1998", "1999", "2000", "2001", "2002", "2003"]
    first_row = True
    n_elements = 0
    
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
                    n_elements += 1

    # Remove outliers
    records.pop(52)
    records.pop(104)
    records.pop(358)
    records.pop(103)
    records.pop(356)
	
    print ""
    print "Total number of elements: " + str(n_elements)
    
    return records, names
# -*- coding: utf-8 -*-

"""

Load the data of San Juan from 1997 to 2003.

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
            # Save the names of the features
            for i in range(4, len(row)):
                names.append(row[i])

            first_row = False
        else:
            # Replace no-data fields with the value 0.
            while ",," in line:
                line = line.replace(",,", ",0,")

            # Replace last unfilled field with the value 0.
            line = line.replace(",\n", ",0\n")

            row = line.split(",")

            # Remove the features which are not useful.
            if row[0] == "sj" and row[1] in years:
                for i in range (4):
                    row.pop(0)
                # Save the row in the records variable.
                if row != []:
                    records.append(map(float, row))

    return records, names

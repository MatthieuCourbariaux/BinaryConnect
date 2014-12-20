
import csv
import re

comp_precision = [19,21,23]
update_precision = [17,19]
initial_range = [5]
dynamic_range = [0]

csv_file = open('X_X_5_0.csv', 'w')
csv_writer = csv.writer(csv_file,  lineterminator = '\n')
csv_writer.writerow(["comp_precision","update_precision", "initial_range","dynamic_range","validation_error","test_error"])

for j in comp_precision:
    for k in update_precision:
        for l in initial_range:
            for m in dynamic_range:
                
                name =  str(j) + "_" + str(k) + "_" + str(l) + "_" + str(m) + ".txt"
                f = open(name, 'r').readlines()
                
                length = len(f)
                
                print f[length-3]
                validation_error = float(re.findall("\d+.\d+", f[length-3])[0])/100.
                
                print f[length-2]
                test_error = float(re.findall("\d+.\d+", f[length-2])[0])/100.
                
                # print f[length-3-44-1]
                # validation_error = float(re.findall("\d+.\d+", f[length-3-44-1])[0])/100.
                
                # print f[length-2-44-1]
                # test_error = float(re.findall("\d+.\d+", f[length-2-44-1])[0])/100.
                
                csv_writer.writerow([j,k,l,m,validation_error,test_error])
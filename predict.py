import csv

def main():
    try : 
        m = 0;
        p = 0;
        with open('weights.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            row = next(reader)   
            m = float(row[0])
            p = float(row[1])
            minimum = float(row[2])
            rng = float(row[3])
        while 1:
            s = input("Enter a number : to get the prediction ") 
            ret = ((float(s) - minimum) / rng)*m + p
            print("the prediction is {}€".format(ret))
    except : pass

if __name__ == "__main__" :
    main();

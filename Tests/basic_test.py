import sys

def return_Arguments():
    if len(sys.argv)>1:
        # Put input variables here
        statement = sys.argv[1]
    else:
        # Put default variables here
        statement = "Hello World"
        
    return statement


def main():
    statement = return_Arguments()
    print(statement)



if __name__ == "__main__":
    main()
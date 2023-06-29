import random

def random_List(data):
    count = 0
    data_list = list(range(1, data+1))
    while count<6:
        count += 1
        number = random.choice(data_list)
        data_list.remove(number)
        print (number)
        
random_List(45)
import time
import turtle
import random

def quit(x,y):  
    turtle.bye()

def fxn(x,y):
    global t0
    
    t= time.time() - t0 
    
    turtle.write(f'반응시간 : {int(t*500)}ms',font=('Arial', 30, 'normal'))
    turtle.onscreenclick(quit)

turtle.onscreenclick(fxn)

time.sleep(random.randrange(1,10))
t0= time.time()
turtle.Screen().bgcolor("orange")


turtle.mainloop()
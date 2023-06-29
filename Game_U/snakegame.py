"""Snake, classic arcade game.

Exercises

1. How do you make the snake faster or slower?
2. How can you make the snake go around the edges?
3. How would you move the food?
4. Change the snake to respond to mouse clicks.
"""

from random import randrange
from turtle import *


def square(x, y, size, name):
    """Draw square at `(x, y)` with side length `size` and fill color `name`.

    The square is oriented so the bottom left corner is at (x, y).

    """
    import turtle

    turtle.up()
    turtle.goto(x, y)
    turtle.down()
    turtle.color(name)
    turtle.begin_fill()

    for count in range(4):
        turtle.forward(size)
        turtle.left(90)

    turtle.end_fill()



food = [0, 0]
food[0] = randrange(-15, 15) * 10
food[1] = randrange(-15, 15) * 10
snake = [[10, 4],[10, 3],[10, 2],[10, 1],[10, 0]]
aim = [0, -10]


def change(x, y):
    """Change snake direction."""
    aim[0] = x
    aim[1] = y


def inside(head):
    """Return True if head inside boundaries."""
    return -200 < head[0] < 190 and -200 < head[1] < 190


def move():
    """Move snake forward one segment."""
    head = snake[-1].copy()
    head[0]+=aim[0]
    head[1]+=aim[1]

    if not inside(head) or head in snake:
        square(head[0], head[1], 9, 'red')
        update()
        return

    snake.append(head)

    #print(snake,head,food)
    if head == food:
        print('Snake:', len(snake))
        food[0] = randrange(-15, 15) * 10
        food[1] = randrange(-15, 15) * 10
    else:
        snake.pop(0)

    clear()

    for body in snake:
        square(body[0], body[1], 9, 'black')

    square(food[0], food[1], 9, 'green')
    update()
    ontimer(move, 100)


setup(420, 420, 370, 0)
hideturtle()
tracer(False)
listen()
onkey(lambda: change(10, 0), 'Right')
onkey(lambda: change(-10, 0), 'Left')
onkey(lambda: change(0, 10), 'Up')
onkey(lambda: change(0, -10), 'Down')
move()
done()
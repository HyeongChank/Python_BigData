1 #Asteroids by @TokyoEdTech / Written in Python 3.5 
2 #Part 0: Finished Demo 
3 
 
4 import os 
5 import random 
6 import time 
7 import math 
8 
 
9 #Import the Turtle module 
10 import turtle 
11 #Set the screensize 
12 turtle.setup(width=800, height=800) 
13 #Required by MacOSX to show the window 
14 turtle.fd(0) 
15 #Set the animations speed to the maximum 
16 turtle.speed(0) 
17 #Change the background color 
18 turtle.bgcolor("black") 
19 #Change the window title 
20 turtle.title("Asteroids") 
21 #Change the background image 
22 #turtle.bgpic("starfield.gif") 
23 #Hide the default turtle 
24 turtle.ht() 
25 #This saves memory 
26 turtle.setundobuffer(1) 
27 #This speeds up drawing 
28 turtle.tracer(0) 
29 
 
30 
 
31 class Shield(turtle.Turtle): 
32     def __init__(self): 
33         turtle.Turtle.__init__(self) 
34         self.speed(0) 
35         self.penup() 
36         self.ht() 
37         self.frame = 1 
38         self.strength = 100 
39      
40     def draw(self): 
41         shield.goto(player.xcor() + 20, player.ycor()) 
42         shield.setheading(90) 
43         shield.pendown() 
44         if shield.strength > 66: 
45             shield.color("green") 
46             shield.pensize(3) 
47         elif shield.strength > 33 and shield.strength <= 66: 
48             shield.color("yellow") 
49             shield.pensize(2) 
50         elif shield.strength > 0 and shield.strength <= 33: 
51             shield.color("red") 
52             shield.pensize(1) 
53         else: 
54             shield.color("black") 
55             shield.strength = 0 
56 
 
57         if self.strength > 0: 
58             shield.circle(20) 
59             shield.penup() 
60             shield.frame += 1 
61         else: 
62             shield.clear() 
63          
64         if shield.frame == 3: 
65             shield.clear() 
66             shield.frame = 1 
67 
 
68 
 
69      
70 
 
71 class Sprite(turtle.Turtle): 
72     def __init__(self, spriteshape, color, startx, starty): 
73         turtle.Turtle.__init__(self, shape = spriteshape) 
74         self.speed(0) 
75         self.penup() 
76         self.color(color) 
77         self.fd(0) 
78         self.goto(startx, starty) 
79         self.speed = 1 
80          
81     def move(self): 
82         self.fd(self.speed) 
83          
84         #Boundary detection 
85         if self.xcor() > 290: 
86             self.setx(self.xcor() - 580) 
87          
88         if self.xcor() < -290: 
89             self.setx(self.xcor() + 580) 
90          
91         if self.ycor() > 290: 
92             self.sety(self.ycor() - 580) 
93          
94         if self.ycor() < -290: 
95             self.sety(self.ycor() + 580) 
96              
97     def is_collision(self, other): 
98         #Distance changes based on shield strength 
99         if shield.strength > 0: 
100             if other.size == 3.0: 
101                 distance = 45 
102             elif other.size == 2.0: 
103                 distance = 35 
104             elif other.size == 1.0: 
105                 distance = 25 
106         else: 
107             if other.size == 3.0: 
108                 distance = 30 
109             elif other.size == 2.0: 
110                 distance = 25 
111             elif other.size == 1.0: 
112                 distance = 20 
113              
114 
 
115         if (self.xcor() >= (other.xcor() - distance)) and \ 
116         (self.xcor() <= (other.xcor() + distance)) and \ 
117         (self.ycor() >= (other.ycor() - distance)) and \ 
118         (self.ycor() <= (other.ycor() + distance)): 
119             return True 
120         else: 
121             return False 
122                  
123 class Player(Sprite): 
124     def __init__(self, spriteshape, color, startx, starty): 
125         Sprite.__init__(self, spriteshape, color, startx, starty) 
126         self.shapesize(stretch_wid=0.6, stretch_len=1.1, outline=None) 
127         self.speed = 0 
128         self.lives = 3 
129         self.shapesize(stretch_wid=0.6, stretch_len=1.1, outline=None) 
130         self.thrust = 1 
131         self.dx = 0 
132         self.dy = 0 
133         self.rotation_speed = 0 
134          
135     def turn_left(self): 
136         self.rotation_speed = 30 
137         h = self.heading() + self.rotation_speed 
138         player.setheading(h) 
139          
140     def turn_right(self): 
141         self.rotation_speed = -30 
142         h = self.heading() + self.rotation_speed 
143         player.setheading(h) 
144                          
145     def accelerate(self): 
146         h = player.heading() 
147         self.dx += math.cos(h*math.pi/180)*self.thrust 
148         self.dy += math.sin(h*math.pi/180)*self.thrust 
149          
150     def hyperspace(self): 
151         #os.system("afplay hyperspace.mp3&") 
152         x = random.randint(-250, 250) 
153         y = random.randint(-250, 250) 
154         self.goto(x, y) 
155         self.dx *= 0.5 
156         self.dy *= 0.5 
157          
158     def move(self):         
159         player.goto(player.xcor()+self.dx, player.ycor()+self.dy)         
160          
161         #Boundary detection 
162         if self.xcor() > 290: 
163             self.setx(self.xcor() - 580) 
164          
165         if self.xcor() < -290: 
166             self.setx(self.xcor() + 580) 
167          
168         if self.ycor() > 290: 
169             self.sety(self.ycor() - 580) 
170          
171         if self.ycor() < -290: 
172             self.sety(self.ycor() + 580) 
173 
 
174     def collides(self, asteroid):                         
175         if shield.strength > 0: 
176             for particle in particles: 
177                 x = (asteroid.xcor() + player.xcor()) / 2.0 
178                 y = (asteroid.ycor() + player.ycor()) / 2.0 
179                 particle.explode(x, y) 
180             asteroid.destroy() 
181             game.score += 100 
182             shield.strength -= int(10 * asteroid.size) 
183         else: 
184             for particle in particles: 
185                 x = (asteroid.xcor() + player.xcor()) / 2.0 
186                 y = (asteroid.ycor() + player.ycor()) / 2.0 
187                 particle.explode(x, y) 
188             game.score -= 100 
189             game.lives -= 1 
190              
191             #Check if player is out of lives. If so, reset. 
192             if game.lives < 1: 
193                 #Reset game 
194                 for asteroid in asteroids: 
195                     asteroid.size = 1 
196                     asteroid.goto(-1000, -1000) 
197 
 
198                 for asteroid in asteroids:                     
199                     asteroid.destroy() 
200                 #Clear the asteroid list                     
201                 #Python 3.3 and above                 
202                 asteroids.clear() 
203 
 
204                 game.level = 1 
205                 game.lives = 3 
206                 game.score = 0 
207                 shield.strength = 100 
208                 player.goto(0, 0) 
209                 game.start_level() 
210 
 
211             asteroid.destroy() 
212          
213         game.show_status() 
214 
 
215              
216 class Asteroid(Sprite): 
217     def __init__(self, spriteshape, color, size, speed, startx, starty): 
218         Sprite.__init__(self, spriteshape, color, startx, starty) 
219         self.shapesize(stretch_wid=size, stretch_len=size, outline=None) 
220         self.speed = speed 
221         self.size = size 
222         self.setheading(random.randint(0,360)) 
223 
 
224     def move(self): 
225         self.fd(self.speed) 
226 
 
227         if self.size == 3.0: 
228             left = -272 
229             right = 272 
230             top = 272 
231             bottom = -272 
232 
 
233         elif self.size == 2.0: 
234             left = -280 
235             right = 280 
236             top = 289 
237             bottom = -280 
238 
 
239         elif self.size == 1.0: 
240             left = -290 
241             right = 290 
242             top = 290 
243             bottom = -290 
244 
 
245 
 
246         #Boundary detection 
247         if self.xcor() > right: 
248             self.setx(self.xcor() - right * 2) 
249          
250         if self.xcor() < left: 
251             self.setx(self.xcor() + right * 2) 
252          
253         if self.ycor() > top: 
254             self.sety(self.ycor() - top * 2) 
255          
256         if self.ycor() < bottom: 
257             self.sety(self.ycor() + top * 2) 
258 
 
259     def destroy(self): 
260             if self.size == 3.0: 
261                 self.size = 2.0 
262                 self.speed = 5 
263                 #Append a new asteroid 
264                 asteroids.append(Asteroid("circle", "brown",2.0 ,4 , self.xcor(), self.ycor())) 
265 
 
266                 #Change Asteroid Size             
267                 self.shapesize(stretch_wid=asteroid.size, stretch_len=asteroid.size, outline=None) 
268                 #Change Heading 
269                 self.setheading(random.randint(0, 360)) 
270 
 
271 
 
272             elif self.size == 2.0: 
273                 self.size = 1.0 
274                 self.speed = 7 
275                 #Append a new asteroid 
276                 asteroids.append(Asteroid("circle", "brown",1.0 ,5 , self.xcor(), self.ycor())) 
277 
 
278                 #Change Asteroid Size             
279                 self.shapesize(stretch_wid=asteroid.size, stretch_len=asteroid.size, outline=None) 
280                 #Change Heading 
281                 self.setheading(random.randint(0, 360)) 
282 
 
283             elif self.size == 1.0: 
284                 self.goto(1000, 1000) 
285                 self.ht() 
286                 if asteroid in asteroids: 
287                     asteroids.remove(asteroid) 
288 
 
289 
 
290 
 
291      
292 class Missile(Sprite): 
293     def __init__(self, spriteshape, color, startx, starty): 
294         Sprite.__init__(self, spriteshape, color, startx, starty) 
295         self.shapesize(stretch_wid=0.2, stretch_len=0.4, outline=None) 
296         self.speed = 20 
297         self.status = "ready" 
298         self.goto(-1000, 1000) 
299          
300     def fire(self): 
301         if self.status == "ready": 
302             #Play missile sound 
303             #os.system("afplay laser.mp3&") 
304             self.goto(player.xcor(), player.ycor()) 
305             self.setheading(player.heading()) 
306             self.status = "firing" 
307              
308     def move(self): 
309         if self.status == "ready": 
310             self.goto(-1000, 1000) 
311          
312         elif self.status == "firing": 
313             self.fd(self.speed)     
314              
315         #Border check 
316         if self.xcor() < -290 or self.xcor() > 290 or \ 
317             self.ycor()< -290 or self.ycor()> 290: 
318             self.goto(-1000,1000) 
319             self.status = "ready" 
320 
 
321 class Particle(Sprite): 
322     def __init__(self, spriteshape, color, startx, starty): 
323         Sprite.__init__(self, spriteshape, color, startx, starty) 
324         self.shapesize(stretch_wid=0.1, stretch_len=0.1, outline=None) 
325         self.goto(-1000,-1000) 
326         self.frame = 0.0 
327          
328     def explode(self, startx, starty): 
329         self.goto(startx,starty) 
330         self.setheading(random.randint(0,360)) 
331         self.frame = 1.0 
332         self.myspeed = random.randint(2, 10) 
333 
 
334 
 
335 
 
336     def move(self): 
337         if self.frame > 0: 
338             self.fd(self.myspeed) 
339             self.frame += 1.0 
340             self.shapesize(stretch_wid=0.3/self.frame, stretch_len=0.3/self.frame, outline=None) 
341 
 
342         if self.frame > 15.0: 
343             self.frame = 0.0 
344             self.goto(-1000, -1000) 
345 
 
346 class Game(): 
347     def __init__(self): 
348         self.level = 1 
349         self.score = 0 
350         self.state = "playing" 
351         self.pen = turtle.Turtle() 
352         self.lives = 3 
353          
354     def start_level(self): 
355         for i in range(self.level): 
356             asteroids.append(Asteroid("circle", "brown", 3.0, 2, random.randint(-300, 300), random.randint(-300, 300))) 
357 
 
358     def draw_border(self): 
359         #Draw border 
360         self.pen.speed(0) 
361         self.pen.color("white") 
362         self.pen.pensize(3) 
363         self.pen.penup() 
364         self.pen.goto(-300, 300) 
365         self.pen.pendown() 
366         for side in range(4): 
367             self.pen.fd(600) 
368             self.pen.rt(90) 
369         self.pen.penup() 
370         self.pen.ht() 
371         self.pen.pendown() 
372          
373     def show_status(self): 
374         self.pen.undo() 
375         msg = "ASTEROIDS! Level: {}  Score: {}  Lives: {}  Shields: {}".format(self.level, self.score, game.lives, shield.strength) 
376         self.pen.penup() 
377         self.pen.goto(-300, 310) 
378         self.pen.write(msg, font=("Arial", 16, "normal")) 
379 
 
380      
381 
 
382 #Create my sprites 
383 player = Player("triangle", "white", 0, 0) 
384 missile = Missile("triangle", "yellow", 0, 0) 
385 shield = Shield() 
386 
 
387 #Create game object 
388 game = Game() 
389 
 
390 #Draw the game border 
391 game.draw_border() 
392 
 
393 #Show the game status 
394 game.show_status() 
395 
 
396 asteroids =[] 
397 
 
398 #Start the level 
399 game.start_level() 
400 
 
401 particles = [] 
402 for i in range(25): 
403     particles.append(Particle("circle", random.choice(["yellow", "red", "orange"]), 0, 0)) 
404 
 
405 #Keyboard bindings 
406 turtle.onkeypress(player.turn_left, "Left") 
407 turtle.onkeypress(player.turn_right, "Right") 
408 turtle.onkeypress(player.accelerate, "Up") 
409 turtle.onkeypress(player.hyperspace, "Down") 
410 turtle.onkeypress(missile.fire, "space") 
411 turtle.listen() 
412 
 
413 #Main game loop 
414 while True: 
415     turtle.update() 
416     time.sleep(0.02) 
417 
 
418     player.move() 
419     missile.move() 
420      
421     for asteroid in asteroids: 
422         asteroid.move() 
423          
424         #Check for a collision with the player 
425         if player.is_collision(asteroid): 
426             #Play explosion sound 
427             #os.system("afplay explosion.mp3&") 
428             #Do the explosion 
429 
 
430             player.collides(asteroid) 
431              
432         #Check for a collision between the missile and the asteroid 
433         if missile.is_collision(asteroid): 
434             #Play explosion sound 
435             #os.system("afplay explosion.mp3&") 
436 
 
437 
 
438             #Reset the Missile Status     
439             missile.status = "ready" 
440 
 
441             #Increase the score 
442             game.score += 100 
443             game.show_status() 
444             #Do the explosion 
445             for particle in particles: 
446                 x = (asteroid.xcor() + missile.xcor()) / 2.0 
447                 y = (asteroid.ycor() + missile.ycor()) / 2.0 
448                 particle.explode(x, y) 
449 
 
450             #If the asteroid is Large, make it small times 2 
451             asteroid.destroy() 
452 
 
453     for particle in particles: 
454         particle.move() 
455 
 
456     #Shield 
457     shield.draw() 
458 
 
459     #Check for end of level 
460     if len(asteroids) == 0: 
461         game.level += 1 
462         game.start_level() 
463 
 
464     game.show_status() 
465 
 
466 delay = input("Press enter to finish. > ") 

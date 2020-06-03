
'''
# Importaciones fisicas / visuales
'''
import sys

import pygame
from pygame.locals import *
from pygame.color import *
    
import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util
#Posiblemente time 

'''
# Importaciones algoritmo
'''
import numpy as np

'''
# Funciones para crear flechas y fisica
'''

def create_arrow():
    vs = [(-30,0), (0,3), (10,0), (0,-3)]
    mass = 1
    moment = pymunk.moment_for_poly(mass, vs)
    arrow_body = pymunk.Body(mass, moment)

    arrow_shape = pymunk.Poly(arrow_body, vs) #Aqui se relacionan
    arrow_shape.friction = .5
    arrow_shape.collision_type = 1
    return arrow_body, arrow_shape
    
def stick_arrow_to_target(space, arrow_body, target_body, position, flying_arrows):
    pivot_joint = pymunk.PivotJoint(arrow_body, target_body, position)
    phase = target_body.angle - arrow_body.angle 
    gear_joint = pymunk.GearJoint(arrow_body, target_body, phase, 1)
    space.add(pivot_joint)
    space.add(gear_joint)
    try:
        flying_arrows.remove(arrow_body)
        # print("Arrow Collide in {}".format(arrow_body.position))
    except:
        pass

def post_solve_arrow_hit(arbiter, space, data):
    if arbiter.total_impulse.length > 300:
        a,b = arbiter.shapes
        position = arbiter.contact_point_set.points[0].point_a
        b.collision_type = 0
        b.group = 1
        other_body = a.body
        arrow_body = b.body
        # print("arrow: {} , to: {}".format(arrow_body, other_body))
        # if data["target"] == other_body:
        #     print("Le diste")
        space.add_post_step_callback(
            stick_arrow_to_target, arrow_body, other_body, position, data["flying_arrows"])

'''
# Funciones del algoritmo
'''
def poblacion_inicial(n):
    power = np.random.randint(0,400,(n,1))
    angle = np.random.rand(n,1)
    angle = np.interp(angle, (0, +1), (-3, 3))
    genotipo = np.concatenate((power, angle),axis=1)
    return genotipo

#Dentro de la funcion fitnes puedo hacer los calculos fisicos

def fitness_poblacion(poblacion, space, cannon_body, cannon_shape, flying_arrows): # Checar
    n = len(poblacion)
    fitness = np.zeros((n))
    # aqui se definen los fps
    fps = 60
    dt = 1./fps

    for i in range(n):

        ind = poblacion[i]
        power = float(ind[0])
        angle = float(ind[1])
        impulse = power*Vec2d(1,0)
        impulse.rotate(angle)

        arrow_body, arrow_shape = create_arrow()

        arrow_body.position = cannon_body.position + Vec2d(cannon_shape.radius + 40, 0).rotated(angle)
        arrow_body.angle = angle

        space.add(arrow_shape)
        arrow_body.apply_impulse_at_world_point(impulse, arrow_body.position) 
        space.add(arrow_body)
        flying_arrows.append(arrow_body)

        
        for k in range(100): # 100 pasos son suficientes para cruzar la pantalla
            #aqui va lo de flying arrows que se inclinan
            for flying_arrow in flying_arrows:
                drag_constant = 0.0002
                
                pointing_direction = Vec2d(1,0).rotated(flying_arrow.angle)
                flight_direction = Vec2d(flying_arrow.velocity)
                flight_speed = flight_direction.normalize_return_length()
                dot = flight_direction.dot(pointing_direction)
                # (1-abs(dot)) can be replaced with (1-dot) to make arrows turn 
                # around even when fired straight up. Might not be as accurate, but 
                # maybe look better.
                drag_force_magnitude = (1-abs(dot)) * flight_speed **2 * drag_constant * flying_arrow.mass
                arrow_tail_position = Vec2d(-50, 0).rotated(flying_arrow.angle)
                flying_arrow.apply_impulse_at_world_point(drag_force_magnitude * -flight_direction, arrow_tail_position)
                
                flying_arrow.angular_velocity *= 0.5

            space.step(dt)
        

        # lo de las fuerzas y el espacio

        err = arrow_body.position.get_distance(Vec2d(800, 100))
        space.remove(arrow_shape, arrow_body)
        fitness[i] = err
    return fitness

def buscar_elite(poblacion,fitness):
    fitIndexes = np.argsort(fitness)
    elite = poblacion[fitIndexes[0]]
    eliteFitness = fitness[fitIndexes[0]]
    return elite, eliteFitness

def mutacion(poblacion):
    n = len(poblacion)
    pobv = np.zeros_like(poblacion)

    power = poblacion[:,0]
    power = np.interp(power, (0, 900), (0, +1))

    angle = poblacion[:,1]
    angle = np.interp(angle, (-3, 3), (0, +1))

    # pobi = poblacion interpolada
    pobi = np.concatenate((power.reshape(-1,1),angle.reshape(-1,1)), axis=1)

    for i in range(n):
        #Index 
        i_1 = np.random.randint(n)
        i_2 = np.random.randint(n)
        i_3 = np.random.randint(n)
        #Ind
        xr1 = pobi[i_1]
        xr2 = pobi[i_2]
        xr3 = pobi[i_3]
        #Vector
        F = np.random.uniform(0, 2)
        vi = xr1 + F*(xr2 - xr3)
        pobv[i] = vi

    power = np.clip(pobv[:,0], 0, 1)
    power = np.interp(power, (0, +1), (0, 900))
    power = power.astype(int)

    angle = np.clip(pobv[:,1], 0, 1)
    angle = np.interp(angle, (0, +1), (-3, 3))
    # angle = angle.astype(int)
    
    pobv = np.concatenate((power.reshape(-1,1),angle.reshape(-1,1)), axis=1)
    return pobv

def cruza(poblacion,pobv,Cr):
    upopulation = np.zeros_like(poblacion)
    n = len(upopulation)
    for i in range(n):
        f_cruza = np.random.random()
        m = len(upopulation[i])
        for u in range(m):
            if f_cruza < Cr:
                upopulation[i][u] = pobv[i][u]
            else:
                upopulation[i][u] = poblacion[i][u]
    return upopulation

def seleccion(poblacion, fitness, pobu, fitu):
    n = len(poblacion)
    for i in range(n):
        if fitu[i] < fitness[i]:
            poblacion[i] = pobu[i]
            fitness[i] = fitu[i]
    return poblacion,fitness


'''
# Hacer las inicializaciones de pymunk
'''
width, height = 890,600

pygame.init()
screen = pygame.display.set_mode((width,height)) 
clock = pygame.time.Clock()
running = True
font = pygame.font.SysFont("Arial", 16)

### Physics stuff
space = pymunk.Space()   
space.gravity = 0,-1000
draw_options = pymunk.pygame_util.DrawOptions(screen)

# walls - the left-top-right walls
static= [pymunk.Segment(space.static_body, (50, 50), (50, 550), 5)
            ,pymunk.Segment(space.static_body, (50, 550), (850, 550), 5)
            ,pymunk.Segment(space.static_body, (850, 550), (850, 50), 5)
            ,pymunk.Segment(space.static_body, (50, 50), (850, 50), 5)
            ]  
#Target green
b2 = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
static.append(pymunk.Circle(b2, 15))
b2.position = 800,100

for s in static:
    s.friction = 1.
    s.group = 1
space.add(static)

# "Cannon" that can fire arrows
#Shooter
cannon_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
cannon_shape = pymunk.Circle(cannon_body, 10)
cannon_shape.sensor = True
cannon_shape.color = (255,50,50)
cannon_body.position = 100,100
space.add(cannon_shape)

# arrow_body,arrow_shape = create_arrow()
# space.add(arrow_shape)

#This list is the arrows that will be in the scene
flying_arrows = []    
handler = space.add_collision_handler(0, 1)
handler.data["flying_arrows"] = flying_arrows
handler.data["target"] = b2 
#Here we have de function that manages collisions
#post_solve is when two shapes touching, it need data that in this case are the arrorws
handler.post_solve=post_solve_arrow_hit

poblacion = poblacion_inicial(10)
fitness = fitness_poblacion(poblacion, space, cannon_body, cannon_shape, flying_arrows)
k = 20
i = 0

while i < k and running:
    #mutacion
    pobv = mutacion(poblacion)
    #reproduccion
    pobu = cruza(poblacion, pobv, 0.5)
    #seleccion
    fitu = fitness_poblacion(pobu, space, cannon_body, cannon_shape, flying_arrows)
    poblacion, fitness = seleccion(poblacion, fitness, pobu, fitu)
    #elite
    elite , eliteFitness = buscar_elite(poblacion, fitness)

    stepsCounter = 0

    ind = elite
    power = float(ind[0])
    angle = float(ind[1])
    impulse = power*Vec2d(1,0)
    impulse.rotate(angle)

    arrow_body, arrow_shape = create_arrow()

    arrow_body.position = cannon_body.position + Vec2d(cannon_shape.radius + 40, 0).rotated(angle)
    arrow_body.angle = angle

    space.add(arrow_shape)
    arrow_body.apply_impulse_at_world_point(impulse, arrow_body.position) 
    space.add(arrow_body)
    flying_arrows.append(arrow_body)

    while(stepsCounter < 150 and running):
        for event in pygame.event.get():
            if event.type == QUIT or \
                event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):  
                running = False
        for flying_arrow in flying_arrows:
                drag_constant = 0.0002
                
                pointing_direction = Vec2d(1,0).rotated(flying_arrow.angle)
                flight_direction = Vec2d(flying_arrow.velocity)
                flight_speed = flight_direction.normalize_return_length()
                dot = flight_direction.dot(pointing_direction)
                # (1-abs(dot)) can be replaced with (1-dot) to make arrows turn 
                # around even when fired straight up. Might not be as accurate, but 
                # maybe look better.
                drag_force_magnitude = (1-abs(dot)) * flight_speed **2 * drag_constant * flying_arrow.mass
                arrow_tail_position = Vec2d(-50, 0).rotated(flying_arrow.angle)
                flying_arrow.apply_impulse_at_world_point(drag_force_magnitude * -flight_direction, arrow_tail_position)
                
                flying_arrow.angular_velocity *= 0.5

        screen.fill(pygame.color.THECOLORS["black"])
        
        ### Draw stuff
        space.debug_draw(draw_options)

        pygame.display.flip()
        
        ### Update physics
        fps = 60
        dt = 1./fps
        space.step(dt)

        stepsCounter+= 1
    space.remove(arrow_shape, arrow_body)
    print("i={}, elite = {}, elite fit: {}".format(i, elite, eliteFitness))
    i+=1
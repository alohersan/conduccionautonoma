"""
In this project, we use a visual simulation to train a self-driving car using NEAT
(NeuroEvolution of Augmenting Topologies). The goal is for the car to navigate a
map of the Marina Bay Circuit (Singapore), using radar sensors to perceive the environment.
NEAT evolves neural network topologies and weights over generations.
This allows for complex, adaptive behavior without explicit programming of driving rules,
only following the neural network configuration described on config.txt

The car used in the simulation is a drawing of the McLaren MP4/4, driven by Ayrton Senna.
Both the circuit and car images were designed by me using GIMP.
"""
#Libraries
import math
import pygame
import neat
import sys
import time

# Define some constants
WIDTH, HEIGHT = 800, 600
CAR_SIZE = 15
BORDER_COLOR = (255, 255, 255, 255)
MAX_SPEED = 10

# Define the Car class with drawing, movement, and sensor logic
class Car:
    def __init__(self, sprite):
        self.sprite = pygame.transform.scale(sprite, (CAR_SIZE, CAR_SIZE))
        self.position = [630, 270]
        self.angle = 0
        self.speed = 5
        self.alive = True
        self.distance = 0
        self.radars = []

    #Draw the car
    def draw(self, screen):
        rotated = pygame.transform.rotate(self.sprite, self.angle)
        rect = rotated.get_rect(center=(self.position[0] + CAR_SIZE // 2, self.position[1] + CAR_SIZE // 2))
        screen.blit(rotated, rect.topleft)
        for radar in self.radars:
            pygame.draw.line(screen, (0, 255, 0), self.center(), radar[0], 1)
            pygame.draw.circle(screen, (0, 255, 0), radar[0], 3)
    #Center the car
    def center(self):
        return [self.position[0] + CAR_SIZE / 2, self.position[1] + CAR_SIZE / 2]

    #Update the car around the map if it's alive
    def update(self, map_img):
        if not self.alive:
            return

        cx, cy = int(self.center()[0]), int(self.center()[1])
        if not (0 <= cx < WIDTH and 0 <= cy < HEIGHT) or map_img.get_at((cx, cy)) == BORDER_COLOR:
            self.alive = False
            return


        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed

        self.distance += self.speed
        self.check_collision(map_img)
        self.check_radars(map_img)

    #Kills the car if detects collisions
    def check_collision(self, map_img):
        x, y = int(self.center()[0]), int(self.center()[1])
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            if map_img.get_at((x, y)) == BORDER_COLOR:
                self.alive = False
        else:
            self.alive = False

    #Logic of the radars displayed from the car
    def check_radars(self, map_img):
        self.radars.clear()
        for d in range(-90, 120, 45):
            length = 0
            while length < 150:
                x = int(self.center()[0] + math.cos(math.radians(360 - (self.angle + d))) * length)
                y = int(self.center()[1] + math.sin(math.radians(360 - (self.angle + d))) * length)
                if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
                    break
                if map_img.get_at((x, y)) == BORDER_COLOR:
                    break
                length += 1
            dist = int(math.hypot(x - self.center()[0], y - self.center()[1]))
            self.radars.append([(x, y), dist])

    def get_data(self):
        return [r[1] / 30 for r in self.radars] + [0] * (5 - len(self.radars))

    #Check if the car is still running
    def is_alive(self):
        return self.alive

    #Rewards for cars moving fordward
    def get_reward(self):
        forward_vector = math.cos(math.radians(360 - self.angle))
        return max(0, self.speed * forward_vector)

#Run NEAT simulation with Pygame
def run_simulation(genomes, config):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT F1 Car Simulator")

    # Load and scale car and map images
    car_sprite = pygame.image.load("mclarenmp44S.png").convert()
    map_img = pygame.image.load("marinabay.png").convert()
    map_img = pygame.transform.scale(map_img, (WIDTH, HEIGHT))

    #Store the cars and neural nwtworks
    nets = []
    cars = []

    #Create neural network from genome
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        car = Car(car_sprite)
        nets.append(net)
        cars.append(car)

    #Displays a message so we know what generation is active
    font = pygame.font.SysFont("Times New Roman", 20)
    global_generation = getattr(run_simulation, "generation", 0) + 1
    setattr(run_simulation, "generation", global_generation)
    clock = pygame.time.Clock()
    counter = 0


    screen.blit(map_img, (0, 0))
    for car in cars:
        car.draw(screen)
    pygame.display.flip()
    time.sleep(0.5)

    # Exit program if the window is closed
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        #Logic where each car makes a decision based on its neural network's output.
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                output = nets[i].activate(car.get_data())
                choice = output.index(max(output))
                if choice == 0:
                    car.angle += 10
                elif choice == 1:
                    car.angle -= 10
                elif choice == 2:
                    pass
                else:
                    car.speed = min(car.speed + 1, MAX_SPEED)
                car.update(map_img)
                genomes[i][1].fitness += car.get_reward()

        # End generation when all cars crash or timeout
        if still_alive == 0 or counter > 30 * 40:
            break

        screen.blit(map_img, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)
        screen.blit(font.render(f"Gen: {global_generation} Alive: {still_alive}", True, (0, 0, 0)), (10, 10))
        pygame.display.flip()
        clock.tick(60)
        counter += 1

# Run the NEAT algorithm with the config file
if __name__ == "__main__":
    config_path = "config.txt"
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    #Create the population, print the progress on the console and run for x generations
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.run(run_simulation, 1000)

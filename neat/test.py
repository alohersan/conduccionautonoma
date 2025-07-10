"""
This code is a test used to determine the initial position of the car on the track.
It loads a background image (the map) and draws a static car sprite on top of it.
No movement or collision logic is implemented.
"""
#Library
import pygame

# Create the environment with the track and the car
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
car_img = pygame.image.load("mclarenmp44S.png").convert()
car_img = pygame.transform.scale(car_img, (30, 30))
map_img = pygame.image.load("marinabay.png").convert()
map_img = pygame.transform.scale(map_img, (WIDTH, HEIGHT))
car_pos = [630, 270]

# Main loop flag
running = True
# Main loop to keep the window open and draw the images
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
    # Draw the map (background) on the screen
    screen.blit(map_img, (0, 0))
    # Draw the car at the specified position
    screen.blit(car_img, car_pos)
    # Update the display to show changes
    pygame.display.flip()

# Quit pygame and clean up resources
pygame.quit()

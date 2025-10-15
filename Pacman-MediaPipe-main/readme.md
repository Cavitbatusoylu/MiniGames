# Pacman Game Implementation with Motion Control

## Aim

To develop a fully functional Pacman game featuring:

- Motion control using hand gestures
- Walls and obstructions
- Enemies

## Local Setup

To run the Pacman game on your local machine:

1. **Clone the repository**
   ```bash
   git clone https://github.com/PraneetDixit/Pacman-MediaPipe.git
   cd Pacman-MediaPipe
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the game**
   ```bash
   python pacman.py
   ```
4. **Note**: Sometimes errors may occur due to missing dependencies like TensorFlow required by MediaPipe. In such cases, install the required modules as prompted.

## Tech Stack Used

- **NumPy**
- **OpenCV**
- **MediaPipe**

## Workflow

### Maze Design and Setup

A Pacman-style maze was selected from the web and modified using Photoshop to create a mask. This mask helps identify traversable areas in the game. Basic image processing and thresholding were applied to achieve this.

### Game Grid Representation

A **NumPy array** was used to represent the maze structure, where traversable areas correspond to pixels in the mask that are colored.

### Game Loop

The game runs in a loop generating frames at **10ms intervals**. Positions, directions, and game variables are managed within this loop.

### Player Representation

The player character is drawn as an **animated ellipse**, with the mouth opening effect achieved by modifying the ellipse angle. Movement is controlled using horizontal and vertical velocities, updating positions accordingly.

### Motion Control and Direction Management

To handle movement commands efficiently:

- Any motion command given is stored in a **"next" variable**.
- The last stored command in this variable determines the turning direction when feasible.
- This approach ensures smoother turns without requiring perfect timing.
- To determine if movement is possible in any direction, a subgrid in the direction of motion is evaluated. If all the pixels in that direction are accessible, the movement is performed.

### Food Mechanics

- Food is spawned in a fixed quantity.
- It is ensured that **n** number of food items are always present in the playing area.
- Positions for food are selected randomly, ensuring placement within the playable area.
- Whenever the player enters a predefined range (subgrid) of pixels corresponding to a food item:
  - The food is considered eaten.
  - The score is incremented.
  - A new food item is spawned in a valid location.

### Enemy Mechanics

- Enemies are randomly spawned initially, ensuring they are in a valid area.
- Each enemy is assigned a random initial velocity direction.
- Enemy motion is controlled through the following rules:
  1. When an enemy encounters a wall, it randomly selects another direction and sets its **"next"** variable accordingly (similar to the main character's mechanics).
  2. To prevent enemies from traveling only end-to-end in straight passages, their direction (**"next"** variable) is changed randomly every **250 iterations**, allowing them to take turns and move more uniformly across the grid.
  3. If an enemy gets stuck in a dead end (a block covered by walls on three sides), additional logic ensures that the direction is changed whenever the enemy becomes stationary, allowing it to re-enter the game area.
- Similar to food mechanics, whenever the player enters a predefined range (subgrid) corresponding to an enemy, contact is assumed, and the game is over.

### Hand Gesture Recognition

- A custom MediaPipe model was created using **MediaPipe Model Maker** to recognize four hand gestures: **thumb up, down, left, and right**.
- The model was trained on a custom dataset containing over **300 images** per gesture.
- The **MediaPipe hand gesture recognition module** was integrated into the game to detect gestures in real time.
- Detected gestures are mapped to directional commands and used to set the **"next" variable**, which controls the player's movement.

### Concurrency with Threading

- Since the game loop and MediaPipe detection loop must operate simultaneously, **threading** was implemented.
- This allows the game to process frame updates and gesture detection concurrently without one blocking the other.

---
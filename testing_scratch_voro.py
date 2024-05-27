#VORONOI DIAGRAM WITH TRIANGULATION ,  DIKSDTRA ALGO FOR CONNECTING THE POINT OF VORONOI 

#ISSUE ARISED FOR CONNECTING THE EDGES OF THOSE POINTS EDGES SOMETIMES CROSSES THE OBSTACLE WHEN CONNECTING THE VORONOI POINTS 
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
import heapq


# Function to create "C" shaped obstacle points
def create_c_obstacle():
    c_points = []
    # Bottom horizontal line of "C"
    for x in np.linspace(2, 8, 7):
        c_points.append([x, 2])
    # Top horizontal line of "C"
    for x in np.linspace(2, 8, 7):
        c_points.append([x, 8])
    # Left vertical line of "C"
    for y in np.linspace(2, 8, 7):
        c_points.append([2, y])
    return np.array(c_points)

# Function to check if a point is inside an obstacle
def is_in_obstacle(point, obstacles):
    for obs in obstacles:
        if np.linalg.norm(point - obs) < 0.3:
            return True
    return False

# Compute the centroids of the Voronoi cells using triangulation
def compute_centroids(vor):
    centroids = []
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            if len(polygon) > 2:  # Ignore lines
                tri = Delaunay(polygon)
                for simplex in tri.simplices:
                    simplex_points = np.array([polygon[i] for i in simplex])
                    centroid = simplex_points.mean(axis=0)
                    centroids.append(centroid)
    return np.array(centroids)

# Dijkstra's algorithm for shortest path
def dijkstra(graph, start, goal):
    queue = [(0, start)]
    distances = {start: 0}
    previous = {start: None}

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == goal:
            break

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    # Reconstruct path
    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = previous[current_node]
    path.reverse()
    return path

# Generate obstacle points for C shape
obstacles = create_c_obstacle()

# Generate start and goal points
start = (1, 1)
goal = (9, 9)

# Concatenate start, goal, and obstacle points
points = np.vstack((start, goal, obstacles))

# Compute Voronoi diagram
vor = Voronoi(points)

# Compute centroids
centroids = compute_centroids(vor)

# Filter centroids to only include those not in obstacles
free_centroids = [tuple(c) for c in centroids if not is_in_obstacle(c, obstacles)]
free_centroids.append(start)
free_centroids.append(goal)

# Print centroid coordinates
print("Centroid Coordinates:")
for centroid in free_centroids:
    print(f"({centroid[0]}, {centroid[1]})")

# Create a graph and add the centroids as nodes
graph = {tuple(c): [] for c in free_centroids}

# Add edges between centroids that are within a certain distance
threshold = 3.2
for i in range(len(free_centroids)):
    for j in range(i + 1, len(free_centroids)):
        if np.linalg.norm(np.array(free_centroids[i]) - np.array(free_centroids[j])) < threshold:
            distance = np.linalg.norm(np.array(free_centroids[i]) - np.array(free_centroids[j]))
            graph[free_centroids[i]].append((free_centroids[j], distance))
            graph[free_centroids[j]].append((free_centroids[i], distance))

# Find the shortest path using Dijkstra's algorithm
path = dijkstra(graph, tuple(start), tuple(goal))

# Extract the positions for plotting
path_positions = np.array(path)

# Plot Voronoi diagram
fig, ax = plt.subplots()
voronoi_plot_2d(vor, ax=ax)

# Plot centroids
centroids_arr = np.array(free_centroids)
ax.plot(centroids_arr[:, 0], centroids_arr[:, 1], 'bo')

# Plot obstacles
ax.plot(obstacles[:, 0], obstacles[:, 1], 'ro')

# Plot start and goal points
ax.plot(start[0], start[1], 'go', markersize=10)
ax.plot(goal[0], goal[1], 'yo', markersize=10)

# Plot path
ax.plot(path_positions[:, 0], path_positions[:, 1], 'g--')

# Plotting adjustments
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


import numpy as np
import networkx as nx
from networkx.algorithms.connectivity import minimum_node_cut

"""
# 1. Create an empty undirected graph
G = nx.Graph()

# 2. Add nodes
G.add_nodes_from([1, 2])
G.add_edges_from([(1, 2)])

# 3. Add undirected edges

G.add_edges_from([
    (1, 2),
    (1, 3),
    (2, 3)
])


# 4. Print the edges
print("Nodes:", G.nodes())
print("Edges:", G.edges())

# 5. Check if the graph is connected
print("Is the graph connected?", nx.is_connected(G))

num_components = nx.number_connected_components(G)
print("Number of connected components:", num_components)

connectivity = nx.node_connectivity(G)
print("Vertex connectivity of G:", connectivity)
cut_nodes = minimum_node_cut(G)
print("These nodes, if removed, disconnect the graph:", cut_nodes)

G.remove_nodes_from(cut_nodes)
print("Nodes:", G.nodes())
print("Edges:", G.edges())
print("Is the graph connected?", nx.is_connected(G))



def undirected_graph_connectivity_analysis(G):
    gross_connectivity = 0
    num_nodes_all = G.number_of_nodes()
    num_nodes_removed = 0
    num_component = nx.number_connected_components(G)
    while num_component + num_nodes_removed != num_nodes_all and num_nodes_removed != num_nodes_all:
        smallest_cut = 99999
        nodes_to_remove = []
        connected_components = nx.connected_components(G)
        for component_nodes in connected_components:
            subgraph = G.subgraph(component_nodes)
            # if nx.number_connected_components(subgraph) >= 2:
            min_mode_cut_num = nx.node_connectivity(subgraph)
            if min_mode_cut_num < smallest_cut and min_mode_cut_num != 0:
                smallest_cut = min_mode_cut_num
                nodes_to_remove = minimum_node_cut(subgraph)
        if smallest_cut == 99999: break
        G.remove_nodes_from(nodes_to_remove)
        gross_connectivity += np.exp(-0.5 * num_nodes_removed) * (smallest_cut - 1)
        num_nodes_removed += smallest_cut
        gross_connectivity += np.exp(-0.5 * num_nodes_removed)
    for i in range(num_nodes_all - num_nodes_removed):
        gross_connectivity += np.exp(-0.5 * (num_nodes_removed + i))
    return gross_connectivity / num_nodes_all


G = nx.Graph()

G.add_nodes_from([1, 2, 3])

G.add_edges_from([
    (1, 2),
    (2, 3),
    (3, 1)
])
connectivity = undirected_graph_connectivity_analysis(G)
print(connectivity)

"""
import networkx as nx
from copy import deepcopy


def custom_erosion_keep_max_degree_remove_all_if_same(G):
    """
    Erosion step:
      - If all nodes in G have the same degree, remove them all.
      - Otherwise, keep only the nodes that have the max degree, remove the rest.

    Returns:
      (H, removed_data) where
        H is the eroded graph,
        removed_data = {
          "nodes": [...],
          "edges": [...]
        } describing what was removed.
    """
    H = G.copy()

    degrees = dict(H.degree())
    if not degrees:
        return H, None  # Empty graph => nothing to remove

    unique_degs = set(degrees.values())
    if len(unique_degs) == 1:
        # All nodes have the same degree => remove them all
        removed_nodes = list(H.nodes())
        removed_edges = []
        for node in removed_nodes:
            for neighbor in list(H[node]):
                edge = tuple(sorted((node, neighbor)))
                removed_edges.append(edge)
        H.remove_nodes_from(removed_nodes)

        removed_data = {"nodes": removed_nodes, "edges": removed_edges}
        return H, removed_data

    # Otherwise, keep only nodes with the max degree
    max_degree = max(degrees.values())
    keep_nodes = [n for n, deg in degrees.items() if deg == max_degree]

    removed_nodes = []
    removed_edges = []
    for node in list(H.nodes()):
        if node not in keep_nodes:
            for neighbor in list(H[node]):
                edge = tuple(sorted((node, neighbor)))
                removed_edges.append(edge)
            removed_nodes.append(node)

    H.remove_nodes_from(removed_nodes)

    removed_data = {"nodes": removed_nodes, "edges": removed_edges}
    return H, removed_data


def general_dilation(G_current, G_orig, removed_data):
    """
    General dilation step:
      1) If G_current is empty, restore ALL nodes/edges from removed_data (fully undo that erosion).
      2) If G_current is NOT empty, then for each node in G_current, add
         any missing neighbors (and edges) from the original graph G_orig.

    Returns the updated graph (G_dilated).
    """
    G_dilated = G_current.copy()

    if G_dilated.number_of_nodes() == 0:
        # CASE A: Graph is empty => fully restore from removed_data
        if removed_data is not None:
            for node in removed_data["nodes"]:
                G_dilated.add_node(node)
            for (u, v) in removed_data["edges"]:
                G_dilated.add_edge(u, v)
    else:
        # CASE B: Graph is not empty => "grow" by neighbors from G_orig
        for node in list(G_dilated.nodes()):
            if node in G_orig:
                orig_neighbors = list(G_orig[node])
                for nbr in orig_neighbors:
                    if nbr not in G_dilated:
                        G_dilated.add_node(nbr)
                    if G_orig.has_edge(node, nbr):
                        G_dilated.add_edge(node, nbr)

    return G_dilated


def erode_and_dilate_until_vanished(G_orig):
    """
    Main algorithm:
      1) If G_orig is NOT connected at the start, return 0 (an integer).
      2) Otherwise, we keep applying erosion steps until the graph is empty:
         - Let x = the count of erosions so far.
         - After each erosion step producing G_current:

             A) If G_current is EMPTY => treat it as connected, record (x, 0), then STOP.

             B) Else if G_current is CONNECTED => record (x, 0)
                and then CONTINUE to the next erosion step
                (we do NOT stop, because we must run until vanished).

             C) Otherwise (non-empty & not connected) =>
                use a local copy to do 'general_dilation' steps until that local copy
                is connected, counting how many calls are needed.
                Record (x, dilation_steps).
                Then CONTINUE to the next erosion step.

      3) Stop once the graph vanishes. Return the list of (erosion_count, dilation_steps).

    Returns:
      - 0 (integer) if G_orig is initially disconnected.
      - Otherwise, a list of (erosion_count, dilation_steps).
    """
    connectivity = 0
    num_c = 1
    # 1) Check if original is connected
    if G_orig.number_of_nodes() == 0:
        return 0
    if not nx.is_connected(G_orig):
        num_c = nx.number_connected_components(G_orig)
        print('workspace is not connected,penalize partial connectivity')
        # return 0

    # results = []  # list of (erosion_count, dilation_steps)
    G_current = G_orig.copy()  # Working graph
    erosion_stack = []

    erosion_count = 0

    # Keep eroding until we vanish (empty) or can't erode further
    while True:
        # Erode the current graph
        G_next, removed_data = custom_erosion_keep_max_degree_remove_all_if_same(G_current)

        if removed_data is None:
            # No further erosion is possible
            # If it's already empty, we can treat that as connected => (erosion_count, 0)
            # But typically if G_current wasn't empty, we are stable. Let's see:
            if G_current.number_of_nodes() == 0:
                # results.append((erosion_count, 0))  # "vanished" at previous step
                connectivity += np.exp(-0.5 * 0)
            # Otherwise, the shape never vanished but can't be eroded more.
            break

        # We successfully performed an erosion
        erosion_count += 1
        erosion_stack.append(removed_data)
        G_current = G_next

        # ---- Check the new graph after erosion ----

        # (A) If G_current is EMPTY => (erosion_count, 0), stop
        if G_current.number_of_nodes() == 0:
            # results.append((erosion_count, 0))
            connectivity += np.exp(-0.5 * 0)
            break

        # (B) If G_current is CONNECTED => (erosion_count, 0);
        #     but continue to next erosion (since we must vanish eventually)
        if nx.is_connected(G_current):
            # results.append((erosion_count, 0))
            connectivity += np.exp(-0.5 * 0)
            # proceed to next iteration
            continue

        # (C) G_current is non-empty but NOT connected => do local dilation
        local_graph = G_current.copy()
        local_stack = deepcopy(erosion_stack)

        dilation_steps = 0

        while True:
            if local_graph.number_of_nodes() > 0 and nx.is_connected(local_graph):
                # local copy is connected, done
                break

            if local_graph.number_of_nodes() == G_orig.number_of_nodes():
                break

            if local_graph.number_of_nodes() == 0:
                # Must pop from local_stack if available
                if local_stack:
                    rd = local_stack.pop()
                    local_graph = general_dilation(local_graph, G_orig, rd)
                    dilation_steps += 1
                else:
                    # No data left to restore => stuck
                    break
            else:
                # Not connected, do neighbor expansion
                previous_num_edges = nx.number_of_edges(local_graph)
                local_graph = general_dilation(local_graph, G_orig, None)
                if nx.number_of_edges(local_graph) == previous_num_edges:
                    break
                dilation_steps += 1

        # Record (erosion_count, dilation_steps)
        # results.append((erosion_count, dilation_steps))
        connectivity += np.exp(-0.5 * dilation_steps)
        # Then proceed with next erosion step
        # (We do NOT actually apply those dilations to G_current —
        #  it's just a local test to see how many expansions are needed.)

        # If we continue, eventually either we remove everything or can't remove more.
    # print(results)
    return connectivity / (erosion_count * num_c)


def fibonacci_sphere_angles(n):
    """
    Generate spherical coordinates (theta, phi) for points evenly distributed on a sphere using the Fibonacci Sphere Algorithm.

    Args:
    - n (int): Number of points. Must be greater than 1.

    Returns:
    - angles (np.ndarray): Array of shape (n, 2) with (theta, phi) in radians.
        - theta: Azimuthal angle (0 to 2π).
        - phi: Polar angle (-π/2 to π/2).
    """
    if n <= 1:
        raise ValueError("Number of points 'n' must be greater than 1.")

    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    z = np.linspace(1, -1, n)  # Evenly spaced z values from 1 to -1
    phi_angles = np.arcsin(z)  # Polar angles (-π/2 to π/2)
    theta_angles = (2 * np.pi * np.arange(n) / phi) % (2 * np.pi)  # Azimuthal angles (0 to 2π)

    return np.column_stack((theta_angles, phi_angles))


def fibonacci_sphere_distance(n):
    """
    Calculate the average geodesic distance between adjacent points on a Fibonacci sphere.

    Args:
    - n (int): Number of points on the sphere.

    Returns:
    - avg_distance (float): Average distance between adjacent points.
    """
    angles = fibonacci_sphere_angles(n)  # Get the (theta, phi) angles
    total_distance = 0

    for i in range(n - 1):  # Adjacent points in sequence
        theta1, phi1 = angles[i]
        theta2, phi2 = angles[i + 1]

        # Great-circle distance formula
        distance = np.arccos(
            np.sin(phi1) * np.sin(phi2) +
            np.cos(phi1) * np.cos(phi2) * np.cos(theta1 - theta2)
        )
        total_distance += distance

    # Average distance (loop back to the first point for adjacency)
    avg_distance = total_distance / (n - 1)
    return avg_distance


# -------------- DEMO --------------
if __name__ == "__main__":
    # Example #1: A connected graph
    G_demo = nx.Graph()
    G_demo.add_edges_from(
        [(0, 1), (0, 9), (0, 10), (1, 2), (1, 10), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
         (11, 12), (11, 20), (11, 21), (11, 22), (11, 23), (11, 31), (11, 32), (12, 13), (12, 21), (12, 22), (12, 23),
         (12, 24), (12, 32), (13, 14), (13, 23), (13, 24), (13, 25), (14, 15), (14, 24), (14, 26), (15, 16), (15, 25),
         (15, 26), (15, 27), (16, 17), (16, 26), (16, 27), (16, 28), (17, 18), (17, 27), (17, 28), (17, 29), (18, 19),
         (18, 28), (18, 29), (18, 30), (19, 20), (19, 29), (19, 31), (20, 21), (20, 22), (20, 30), (20, 31), (20, 32),
         (21, 22), (21, 23), (21, 31), (21, 32), (22, 23), (22, 31), (22, 32), (23, 24), (23, 32), (24, 25), (25, 26),
         (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (33, 34), (33, 42), (33, 43), (33, 44), (33, 45),
         (33, 53), (33, 54), (34, 35), (34, 43), (34, 44), (34, 45), (34, 46), (34, 54), (35, 36), (35, 45), (35, 46),
         (35, 47), (36, 37), (36, 46), (36, 47), (36, 48), (37, 38), (37, 47), (37, 48), (37, 49), (38, 39), (38, 48),
         (38, 49), (38, 50), (39, 40), (39, 49), (39, 50), (39, 51), (40, 41), (40, 50), (40, 51), (40, 52), (41, 42),
         (41, 51), (41, 52), (41, 53), (42, 43), (42, 44), (42, 52), (42, 53), (42, 54), (43, 44), (43, 45), (43, 53),
         (43, 54), (44, 45), (44, 53), (44, 54), (45, 46), (45, 54), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51),
         (51, 52), (52, 53), (53, 54), (55, 56), (55, 64), (55, 65), (55, 66), (55, 67), (55, 75), (55, 76), (56, 57),
         (56, 65), (56, 66), (56, 67), (56, 68), (56, 76), (57, 58), (57, 67), (57, 68), (57, 69), (58, 59), (58, 68),
         (58, 69), (58, 70), (59, 60), (59, 69), (59, 70), (59, 71), (60, 61), (60, 70), (60, 71), (60, 72), (61, 62),
         (61, 71), (61, 72), (61, 73), (62, 63), (62, 72), (62, 73), (62, 74), (63, 64), (63, 73), (63, 74), (63, 75),
         (64, 65), (64, 66), (64, 74), (64, 75), (64, 76), (65, 66), (65, 67), (65, 75), (65, 76), (66, 67), (66, 75),
         (66, 76), (67, 68), (67, 76), (68, 69), (69, 70), (70, 71), (71, 72), (72, 73), (73, 74), (74, 75), (75, 76),
         (77, 78), (77, 86), (77, 87), (77, 88), (77, 89), (77, 97), (77, 98), (78, 79), (78, 87), (78, 88), (78, 89),
         (78, 90), (78, 98), (79, 80), (79, 89), (79, 90), (79, 91), (80, 81), (80, 90), (80, 91), (80, 92), (81, 82),
         (81, 91), (81, 92), (81, 93), (82, 83), (82, 92), (82, 93), (82, 94), (83, 84), (83, 93), (83, 94), (83, 95),
         (84, 85), (84, 94), (84, 95), (84, 96), (85, 86), (85, 95), (85, 96), (85, 97), (86, 87), (86, 88), (86, 96),
         (86, 97), (86, 98), (87, 88), (87, 89), (87, 97), (87, 98), (88, 89), (88, 97), (88, 98), (89, 90), (89, 98),
         (90, 91), (91, 92), (92, 93), (93, 94), (94, 95), (95, 96), (96, 97), (97, 98), (99, 100), (99, 108),
         (99, 109), (99, 110), (99, 111), (99, 119), (99, 120), (100, 101), (100, 109), (100, 110), (100, 111),
         (100, 112), (100, 120), (101, 102), (101, 111), (101, 112), (101, 113), (102, 103), (102, 112), (102, 113),
         (102, 114), (103, 104), (103, 113), (103, 114), (103, 115), (104, 105), (104, 114), (104, 115), (104, 116),
         (105, 106), (105, 115), (105, 116), (105, 117), (106, 107), (106, 116), (106, 117), (106, 118), (107, 108),
         (107, 117), (107, 118), (107, 119), (108, 109), (108, 110), (108, 118), (108, 119), (108, 120), (109, 110),
         (109, 111), (109, 119), (109, 120), (110, 111), (110, 119), (110, 120), (111, 112), (111, 120), (112, 113),
         (113, 114), (114, 115), (115, 116), (116, 117), (117, 118), (118, 119), (119, 120)]
    )
    print("=== Connected Graph Demo ===")
    output_connected = erode_and_dilate_until_vanished(G_demo)
    print("Result:", output_connected)

    # Example #2: A disconnected graph => returns 0 immediately
    G_disc = nx.Graph()
    G_disc.add_edges_from([(1, 2), (3, 4)])  # 2 components
    print("\n=== Disconnected Graph Demo ===")
    output_disc = erode_and_dilate_until_vanished(G_disc)
    print("Result:", output_disc)

# ----------------------- DEMO -----------------------

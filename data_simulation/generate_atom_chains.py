import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

np.seterr(all = "raise") 

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--number", default=10, help="# of samples to generate", type=int)
    parser.add_argument("-p", "--points", default=100, help="# points per sample", type=int)
    
    parser.add_argument("--mode", choices=["distance", "energy"], default="energy")
    parser.add_argument("--radius", type=int, default=5)
    
    parser.add_argument("-o", "--out", type=Path, default=Path.cwd())

    return parser


def plot_chain(coordinates, charges, outdir:Path):
    coord = pd.DataFrame(coordinates, columns=["X", "Y", "Z"])
    charge = pd.DataFrame(charges, columns=["Charge"])
    df = pd.concat([coord, charge], axis=1)
    df["Number"] = list((range(charges.shape[0])))[::-1]
    
    fig = go.Figure(data=go.Scatter3d(
            x=df["X"], y=df["Y"], z=df["Z"],
            marker=dict(
                size=4,
                color=df["Number"],
                colorscale='Plasma',
                opacity=0.9,
            ),
            line=dict(
                color='rgb(189,189,189)',
                width=1,
            )
        ))
    
    fig.update_layout(
            width=800,
            height=700,
            autosize=False,
            scene=dict(
                camera=dict(
                    up=dict(
                        x=0,
                        y=0,
                        z=1
                    ),
                    eye=dict(
                        x=0,
                        y=1.0707,
                        z=1,
                    )
                ),
                aspectratio = dict( x=1, y=1, z=0.7 ),
                aspectmode = 'manual'
            ),
            showlegend=True
        )
            
    outname = str(outdir / "typical_chain.html")
    print(f"Plotting typical chain and saving to {outname}")
    fig.write_html(outname)
    
    
def plot_histogram(labels):
    fig = px.histogram(labels, )
    pass

def generate_point_in_sphere_with_shift(origin, radius):

    unit = np.random.uniform(low=0, high=1)
    multiplier = radius * np.cbrt(unit)
    
    std_normal_xyz = np.random.standard_normal(size=3)
    length_normalization_multiplier = 1 / np.sqrt(std_normal_xyz @ std_normal_xyz) # normalise by length
    
    point_inside_the_sphere_without_shift = multiplier * length_normalization_multiplier * std_normal_xyz
    
    point_shifted_from_origin = np.around(point_inside_the_sphere_without_shift + origin, 2)
    
    
    return point_shifted_from_origin
    


def compute_charge_energy_between_two_points(xyz_1, xyz_2, charge_1, charge_2):
    distance_vector = xyz_1 - xyz_2
    #/////print(distance_vector)
    return round(charge_1 * charge_2 / np.sqrt(distance_vector @ distance_vector), 4)
 
        
def create_synthetic_chain(points, radius):
    
    #interdot_radius = np.random.exponential(scale=5, size=points-1) + min_radius # lambda = 1/2, shifted distribution
    #interdot_radius = np.random.uniform(low=min_radius, high=max_radius, size=points-1)
    
    coordinates = [np.zeros(3)]
    
    for loop in range(points - 1):
        
        while sum(xyz := generate_point_in_sphere_with_shift(origin=coordinates[-1], radius=radius) - coordinates[-1]) == 0:
            continue
        
        coordinates.append(xyz)

        
    # generating charges
    positive_charges = set(np.random.choice(list(range(points)), points // 2, replace=False))
    charges = np.array([index in positive_charges for index in range(points)], dtype=int)
    
    coordinates = np.around(np.array(coordinates), decimals=4)
    
    return coordinates, charges


def parameterize_chain(coordinates, charges, mode):
    interatom_energies = np.zeros(shape=(coordinates.shape[0], coordinates.shape[0]))
    
    for i, coord_i in enumerate(coordinates):
        for j_ in range(i+1, coordinates.shape[0]):
            j = j_
            coord_j = coordinates[j]
            

            charge_energy = compute_charge_energy_between_two_points(xyz_1=coord_i,
                                                                     xyz_2=coord_j,
                                                                     charge_1=charges[i],
                                                                     charge_2=charges[j])
            interatom_energies[i, j] = interatom_energies[j, i] = charge_energy
            
    
    upper_triangular_indices = np.triu_indices_from(interatom_energies)
    
    potential_electrostatic_energy = interatom_energies[upper_triangular_indices].sum()
    
    if mode =="energy":
        return interatom_energies, potential_electrostatic_energy
    
    distance_matrix = np.zeros(shape=(coordinates.shape[0], coordinates.shape[0]))
    # TODO
    
    
    
def main():
    parser = get_parser()
    args = parser.parse_args()
    
    
    radius = args.radius
    outdir = args.out / f"chains_mode_{args.mode}_radius_{radius}_size_{args.number}" / "raw"
    
    outdir.mkdir(parents=True, exist_ok=False)
    
    
    energies = []
    for i in tqdm(range(1, args.number+1), desc="Parameterizing atom chains and saving to matrix_coordinates_files"):
        
        interatom_energies, total_energy = parameterize_chain(*create_synthetic_chain(points=args.points,
                                                                                      radius=radius,
                                                                                      ),
                                                              mode=args.mode)
        
        energies.append(total_energy)
        
        with open(outdir / f"X_{i}.npy", "wb") as f:
            np.save(f, interatom_energies)
        
    
    with open(outdir / f"Y.npy", "wb") as f:
        np.save(f, np.array(energies))
    

    coord, charges = create_synthetic_chain(points=args.points, radius=radius)
    plot_chain(coordinates=coord, charges=charges, outdir=outdir)
    
    
    
        
if __name__ == "__main__":
    main()
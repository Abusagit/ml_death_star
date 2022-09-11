import numpy as np
import argparse
from pathlib import Path
import tqdm

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--number", default=10000, help="# of samples to generate", type=int)
    parser.add_argument("-p", "--points", default=100, help="# points per sample", type=int)
    parser.add_argument("-r", "--radius", default=50, help="Radius of a sphere containing dots", type=int)
    parser.add_argument("--one_hot_charges", "--one_hot", action="store_true", required=False, 
                        help="If given, charges will be represented as 0/1 encoding, indicating 0 as NEGATIVE charge and 1 as POSITIVE")
    
    parser.add_argument("--range", nargs=2, type=int, default=)
    
    parser.add_argument("-o", "--out", default=Path(Path.cwd()))

    return parser


class SphereDataset:
    def __init__(self, size, amount, onehot_charges, outdir, sphere_radius, normalise=None) -> None:
        self.size = size
        self.onehot_charges = bool(onehot_charges)
        self.outdir = Path(outdir, f"spheres_number_{size}_radius_{sphere_radius}_points_{amount}{'_onehot_charge' * bool(onehot_charges)}", "raw")
        self.radius = sphere_radius
        self.amount = amount

    @staticmethod
    def normalize_vector_by_length(matrix):
        return np.sqrt(np.array([1 / (x @ x) for x in matrix]))

    def generate_charges(self):
        positive_charges = set(np.random.choice(list(range(self.amount)), self.amount // 2, replace=False))

        # print(positive_charges)
        charges = np.array([index in positive_charges for index in range(self.amount)], dtype=int)

        if not self.onehot_charges:
            charges = charges * 2 - 1

        return charges

    def generate_points_inside_sphere(self):

        unit = np.random.uniform(low=0, high=1, size=self.amount)

        multiplier = self.radius * np.cbrt(unit)

        iid_std_normals_3_x_n = np.random.standard_normal(size=(self.amount, 3))
        length_normalization_factor = self.normalize_vector_by_length(iid_std_normals_3_x_n)

        s = multiplier * length_normalization_factor
        scalar_elementwice = np.c_[s, s, s]
        return np.around(scalar_elementwice * iid_std_normals_3_x_n, decimals=4)
    
    def compute_values_for_molecules(self, coordinates, charges):
        sphere_charges = np.zeros((coordinates.shape[0], coordinates.shape[0]))
        ones = np.ones(coordinates.shape[0])

        if self.onehot_charges:
            charges = (charges * 2) - 1

        for i in range(coordinates.shape[0]):
            q_1 = charges[i]

            for j in range(i + 1, coordinates.shape[0]):

                q_2 = charges[j]
                
                distance_vector = coordinates[i] - coordinates[j]

                sphere_charges[i, j] = q_1 * q_2 / np.sqrt(distance_vector @ distance_vector)


        charge = ones @ sphere_charges @ ones
        # breakpoint()

        return round(charge, 4)

    def generate_samples(self):
        Path(self.outdir).mkdir(parents=True, exist_ok=True)
        path = str(Path(self.outdir))
        print(path)
        values = []
        for i in tqdm.tqdm(range(1, self.size+1), desc="Generating points inside sphere and computing ground truth values"):
            charges = self.generate_charges()
            point_coordinates = self.generate_points_inside_sphere() # / self.radius # normalisation
            koulomb_value = self.compute_values_for_molecules(point_coordinates, charges)
            
            values.append(koulomb_value)

            np.savetxt(path + f"/X_{i}_radius_{self.radius}_points_{self.amount}.csv", np.c_[point_coordinates, charges], delimiter=',', fmt='%1.4f,' * 3 + '%d')
        np.savetxt(path + f"/Y_radius_{self.radius}_points_{self.amount}.csv", np.array(values), delimiter=',', fmt='%1.4f')
            

            


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    spheres = SphereDataset(size=args.number,
                            onehot_charges=args.one_hot_charges,
                            outdir=args.out,
                            sphere_radius=args.radius,
                            amount=args.points)

    spheres.generate_samples()

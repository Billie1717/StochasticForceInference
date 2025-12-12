
"""
Converted script from Jupyter Notebook.
Modularised, CLI-enabled, and saves resultsFile2 output.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
from random import randrange
from scipy import linalg    
import seaborn as sns
from scipy.stats import linregress
from scipy.spatial import Delaunay
from collections import defaultdict
import os
import pickle

warnings.filterwarnings('ignore')

def laguerre_polynomialsExp(r,NumBasisFunctions,r_mod):
        # Initialize Laguerre*exp(-r) and their derivatives
        L_exp = [np.exp(-r/r_mod), (1 - r) * np.exp(-r/r_mod)]
        dL_exp = [-np.exp(-r/r_mod), (r - 2) * np.exp(-r/r_mod)]
    
        # Recursively build L_k(r) * exp(-r)
        L_plain = [1.0, 1 - r]  # plain Laguerre polynomials L_k(r)
    
        for k in range(2, NumBasisFunctions):
            # Compute next Laguerre polynomial
            Lk = ((2*k - 1 - r) * L_plain[k-1] - (k - 1) * L_plain[k-2]) / k
            L_plain.append(Lk)
    
            # Multiply by exp(-r)
            Lk_exp = Lk * np.exp(-r/r_mod)
            L_exp.append(Lk_exp)
    
            # Compute the derivative of L_k(r) * exp(-r)
            Dlk = -Lk_exp*np.exp(-r/r_mod) + np.exp(-r/r_mod) * (1/r)* (k*Lk-k*L_plain[k-1])
            dL_exp.append(Dlk)
    
        #print(np.shape(np.reshape(L_exp,5)))
        return np.array(L_exp, dtype=np.float64) #L_exp #, dL_exp
    
def build_triangulation_neighbors(df):
    """
    Builds a neighbor map from Delaunay triangulation of 2D points.

    Parameters:
        df (pd.DataFrame): must contain 'x', 'y', and 'id' columns.

    Returns:
        neighbors (dict): maps id -> set of connected neighbor ids.
        tri (Delaunay object): the triangulation object (optional, useful for plotting).
    """
    coords = df[['x', 'y']].values
    ids = df['id'].values
    if len(coords) < 3:
        print("Too few points for triangulation.",len(coords))
        return 0
    #print("Building triangulation for", len(coords), "points.")
    tri = Delaunay(coords)

    # Map triangle vertex indices to actual IDs
    neighbors = defaultdict(set)

    for simplex in tri.simplices:  # Each simplex is a triangle of indices into coords
        id0, id1, id2 = ids[simplex[0]], ids[simplex[1]], ids[simplex[2]]
        neighbors[id0].update([id1, id2])
        neighbors[id1].update([id0, id2])
        neighbors[id2].update([id0, id1])

    return dict(neighbors)
    
def Get_Distance_Quantities(df_atoms):
    timesteps = df_atoms['timestep'].unique()
    df_atoms = df_atoms.sort_values(by=["id", "timestep"]).copy()
    #NumAtoms = np.max(df_atoms.id)
    r_IJav = []
    AvNumNeighs = []
    R_n1 = []
    R_all = []
    Triangulate = False
    for t in timesteps:
        if t%(len(timesteps)//10) == 0:
            print(f"Processing timestep {t}/{timesteps[-1]}")
        df_t = df_atoms[df_atoms['timestep'] == t]
        if len(df_t) == 1:
            #print(f"Skipping timestep {t} due there only being one cell present")
            continue  # skip incomplete timesteps
        coords = df_t.sort_values('id')[['x', 'y']].values  # ensure ordered by id
        if len(coords) > 3:
            Triangulate = True
        else:
            Triangulate = False

        coordsID = df_t.sort_values('id')[['x', 'y','id']]

        NumAtomsT = len(coords)
        frame1 = df_atoms[df_atoms['timestep'] == t].sort_values('id')
    
        if Triangulate:
            #print("len(coords):", len(coords), "NumAtomsT:", NumAtomsT)
            neighbours=build_triangulation_neighbors(coordsID)
        for i in range(NumAtomsT):
            if Triangulate:
                #print("len(coords):", len(coords), "i:", i, "NumAtomsT:", NumAtomsT)
                NeighIDs = neighbours.get(i+1, set())  # +1 because ids are 1-based in the DataFrame
                NeighIDs = [n for n in NeighIDs if n <= NumAtomsT]
                JAtoms = len(NeighIDs)
                AvNumNeighs.append(JAtoms)
            #else:
            #    NeighIDs = [j+1 for j in range(NumAtomsT) if j != i]

            closest_r = 0
            for j in range(i+1,NumAtomsT):
                #j = NeighIDs[jj]-1
                dxij = coords[j, 0] - coords[i, 0]
                dyij = coords[j, 1] - coords[i, 1] 
                rij = np.sqrt(dxij**2 + dyij**2)
                R_all.append(rij)
                if j==i+1:
                    closest_r = rij
                elif rij < closest_r:
                    closest_r = rij
                if Triangulate:
                    if j in NeighIDs:
                        r_IJav.append(rij)
                else:
                    r_IJav.append(rij)

            R_n1.append(closest_r)
    #bin or hist R_all
    bin_width = (max(R_all) - min(R_all)) / 50  # Adjust bin width as needed
    bins = np.arange(min(R_all), max(R_all) + bin_width, bin_width)
    indices = np.digitize(R_all, bins)
    counts = np.bincount(indices)
    # Find mode bin (bin with highest count)
    mode_bin_index = np.argmax(counts)
    mode_bin_range = (bins[mode_bin_index - 1], bins[mode_bin_index])  # bin edges
    return np.mean(r_IJav),np.mean(R_n1),mode_bin_range,np.mean(AvNumNeighs)
    
def check_cs(df_atoms, Bm05,NumBasisFunctions,r_mod,rcut_inf,rcut_yes):
    # Initialize a matrix to store the time-averaged product of c_alpha and c_beta
    timesteps = df_atoms['timestep'].unique()
    #D = NumBasisFunctions
    df_atoms = df_atoms.sort_values(by=["id", "timestep"]).copy()
    #B_accum = np.zeros((D, D))
    c_alpha_beta_sum = np.zeros((NumBasisFunctions, NumBasisFunctions))
    NumAtoms = np.max(df_atoms.id)
    count = 0
    count = 0
    for t in timesteps:
        #print("Number of atoms:", NumAtoms,len)
        df_t = df_atoms[df_atoms['timestep'] == t]
        if len(df_t) == 1:
            #print(f"Skipping timestep {t} due there only being one cell present")
            continue  # skip incomplete timesteps
        coords = df_t.sort_values('id')[['x', 'y']].values  # ensure ordered by id
        NumAtomsT = len(coords)
        for i in range(NumAtomsT):
            for j in range(i+1,NumAtomsT):
                dx = coords[j, 0] - coords[i, 0] ## dx direction is towards atom j
                dy = coords[j, 1] - coords[i, 1]
                rij = np.sqrt(dx**2 + dy**2)
                if rcut_yes == True:
                    if rij > rcut_inf:
                        continue
                count += 1
                b = laguerre_polynomialsExp(rij,NumBasisFunctions,r_mod)
                c = Bm05.dot(b)
                c_alpha_beta_sum += np.outer(c, c)  # Outer product of c with itself
    # Compute the time average
    c_alpha_beta_avg = c_alpha_beta_sum / (count)

    print("<c_alpha.c_beta> should be Identity matrix:\n", c_alpha_beta_avg)
    
def estmateforcesMultiPairLJ(df_atoms,NumBasisFunctions,OnlyNN,rcut_inf,rcut_yes,r_mod,timestep_skip):
    # Initialize lists to store result
    timesteps = df_atoms['timestep'].unique()
    timesteps = np.sort(timesteps)  # Ensure timesteps are sorted
    #skip every other timestep
    timestepsS = timesteps[::timestep_skip]

    counterskip = 0
    couterNoskip = 0
    D = NumBasisFunctions
    df_atoms = df_atoms.sort_values(by=["id", "timestep"]).copy()
    B_accum = np.zeros((D, D), dtype=np.float64)
    count = 0
    print("creating B matrix...")
    print("Number of timesteps:", len(timesteps))
    for t in timestepsS:
        if t%(len(timestepsS)//10) == 0:
            print(f"Processing timestep {t}/{timestepsS[-1]}")
        #print("Number of atoms:", NumAtoms,len)
        df_t = df_atoms[df_atoms['timestep'] == t]
        #print IDs of cells in this timestep
        #print("IDs in timestep", t, ":", df_t['id'].unique())
        if len(df_t) == 1:
            #print(f"Skipping timestep {t} due there only being one cell present")
            counterskip+=1
            continue  # skip incomplete timesteps
        couterNoskip +=1
        coords = df_t.sort_values('id')[['x', 'y']].values  # ensure ordered by id
        NumAtomsT = len(coords)
        #fetch the index of the atoms in this timestep

        for i in range(NumAtomsT):
            for j in range(i+1,NumAtomsT):
                dx = coords[j, 0] - coords[i, 0] ## dx direction is towards atom j
                dy = coords[j, 1] - coords[i, 1]
                rij = np.sqrt(dx**2 + dy**2)
                if rcut_yes == True:
                    if rij > rcut_inf:
                        continue
                basis_vec = laguerre_polynomialsExp(rij,NumBasisFunctions,r_mod)
                B_accum += np.outer(basis_vec, basis_vec) #for now only checking if pair is present at time t not t+1, may need to add this 
                B_accum += np.outer(basis_vec, basis_vec)
                count += 2
    if count == 0:
        print("No valid timesteps found.")
        return None
    # Average over timesteps
    B = B_accum / count
    #if np.linalg.cond(B) > 1e10:
    #    raise ValueError("Matrix B is ill-conditioned.")
    print("shape of B",np.shape(B))
    
    Bm1 = linalg.inv(B) 
    Bm1 = np.array(Bm1, dtype=np.float64)
    Bm1 = np.array(Bm1, dtype=np.float64, copy=True)
    Bm05 = linalg.sqrtm(Bm1)
    check_cs(df_atoms,Bm05,NumBasisFunctions,r_mod,rcut_inf,rcut_yes)
    print("Inferring forces...")
    FF_accum = np.zeros((1,D)) ##these will be dotted at the end
    Rhist = []
    Lentraj = 0
    for t_idx in range(1,len(timestepsS) - 2):
        if t_idx%(len(timestepsS)//10) == 0:
            print(f"Processing timestep pair {t_idx}/{len(timestepsS) - 1}")
        
        # Correction means X is calculated as the mean of 3 timepoints
        # And velocity is calculated symmetrically.
        # For pairwise 

        t1, t2, t3 = timestepsS[t_idx-1], timestepsS[t_idx], timestepsS[t_idx + 1]
        frame1 = df_atoms[df_atoms['timestep'] == t1].sort_values('id')
        frame2 = df_atoms[df_atoms['timestep'] == t2].sort_values('id')
        frame3 = df_atoms[df_atoms['timestep'] == t3].sort_values('id')
        # Get IDs present in both frames
        #common_ids = np.intersect1d(frame1['id'].values, frame2['id'].values)
        common_ids0 = np.intersect1d(frame1['id'].values, frame3['id'].values) 
        common_ids = np.intersect1d(common_ids0, frame2['id'].values)
        #print("number of common atoms between frames:",len(common_ids))
        #print("at time",t_idx,"example common ids:",common_ids0[:10],"frame3",frame2['id'].values[:10])
        if len(common_ids) == 0:
            #print(f"No common atoms between timesteps {t1} and {t2}. Skipping this pair.")
            continue
        # Filter to only those atoms
        #print("number of atoms in frame3 b4 isin:",len(frame3))
        frame1 = frame1[frame1['id'].isin(common_ids)].sort_values('id')
        frame2 = frame2[frame2['id'].isin(common_ids)].sort_values('id')
        frame3 = frame3[frame3['id'].isin(common_ids)].sort_values('id')
        #print(f"Processing {len(common_ids)} common atoms between timesteps {t1}, {t2} and {t3}.")
        #print("number of atoms in frame3:",len(frame3))
        # Get aligned position arrays
        pos1 = frame1[['x', 'y']].values
        pos2 = frame2[['x', 'y']].values
        pos3 = frame3[['x', 'y']].values
        ddt = float(t2 - t1)
        NumAtomsT = len(common_ids)

        

        ddt = float((t2 - t1))
        #NumAtomsT = len(coords)
        for i in range(NumAtomsT):
            dxi = (pos3[i, 0] - pos1[i, 0])/2
            dyi = (pos3[i, 1] - pos1[i, 1])/2
            velsvec_i = [dxi/ ddt,dyi/ ddt]  #velocity vector of particle i

            for j in range(i+1,NumAtomsT):
                dxj = (pos3[j, 0] - pos1[j, 0])/2
                dyj = (pos3[j, 1] - pos1[j, 1])/2

                if OnlyNN==True:
                    if np.sqrt(dxj**2 + dyj**2) > rcut_inf:
                        continue
                velsvec_j = [dxj/ ddt,dyj/ ddt]  #velocity vector of particle j

                dxij = (1/3)*(pos1[j, 0]+pos2[j, 0]+pos3[j, 0]) - (1/3)*(pos1[i, 0]+pos2[i, 0]+pos3[i, 0])
                dyij = (1/3)*(pos1[j,1]+pos2[j,1]+pos3[j,1]) - (1/3)*(pos1[i,1]+pos2[i,1]+pos3[i,1])

                rij = np.sqrt(dxij**2 + dyij**2) #the distance between particles i and j at start time point t1
                Rhist.append(float(rij))
                if rcut_yes == True:
                    if rij > rcut_inf:
                        continue

                #Normalised r_ij vector
                r_ijVec = np.array([dxij,dyij])/rij 
                velsvec_Parali= np.dot(velsvec_i,-r_ijVec) # projection of velocity i onto the vector connecting i and j
                #velsvec_Perpi = velsvec_i-velsvec_Parali
                velsvec_Paralj= np.dot(velsvec_j,r_ijVec) # j dotted with vector j->i
                #velsvec_Perpj = velsvec_j-velsvec_Paralj

                #Now calculate the F_r_alpha =DeltaX*C_alpha for this pair at this time
                phi = laguerre_polynomialsExp(rij,NumBasisFunctions,r_mod)
                c = Bm05 @ phi
                if np.iscomplexobj(c): ## hmm should find out why it would be complex though...
                    c = c.real
                FF_accum += np.outer(velsvec_Parali,c)
                FF_accum += np.outer(velsvec_Paralj,c)
                Lentraj += 2

    #print("dimension of velsAv",np.shape(velsvec))
    print("dimension of FF_accum",np.shape(FF_accum))
    FF = FF_accum/Lentraj

    return FF, Rhist,Bm05
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run pair inference analysis.")
    parser.add_argument("--FileName", type=str, default="mydata.csv")
    parser.add_argument("--r_range", type=float, default=100)
    parser.add_argument("--NumBasisFunctions", type=int, default=10)
    parser.add_argument("--rcut_inf", type=float, default=None)
    parser.add_argument("--rcut_yes", 
                   type=lambda x: x.lower() == "true", 
                   default=False,
                   help="Enable rcut limit (use 'True' or 'False')")
    parser.add_argument("--r_mod", type=float, default=4)
    parser.add_argument("--timestep_skip", type=int, default=1)
    parser.add_argument("--output_file", type=str, default="resultsFile2.csv")
    return parser.parse_args()

def get_timestepHistogram(Cell_df,CellInd):
    TrajLengths = []
    TrajLengthsTime = []
    TrajSteps = []
    counterN25 = 0
    counter25 = 0
    timestepsAll = Cell_df['timestep']
    timestepsUnique = Cell_df['timestep'].unique()
    print(len(timestepsAll),len(timestepsUnique))
    for cell in range(len(CellInd)-1): #
        #print("Cell", cell,'round',ii)
        ci = cell
        Tstart = CellInd[ci] + 1
        Tend = CellInd[ci + 1]
        trajectory_data = Cell_df.iloc[Tstart:Tend] #.sort_values(by='timestep') #Cell_df.iloc[Tstart:Tend]
        TrajLengths.append(len(trajectory_data))
        TrajLengthsTime.append(trajectory_data['timestep'].max() - trajectory_data['timestep'].min())

        for i in range(len(trajectory_data)):
            if i < len(trajectory_data) - 1:
                t1 = trajectory_data.iloc[i]['timestep']
                t2 = trajectory_data.iloc[i + 1]['timestep']
                step = t2 - t1
                TrajSteps.append(step)
            #print("Cell", cell, "Step size:", step,t1,t2)
            if step != 0.25: #np.round(step,2)
                counterN25 +=1
            else:
                counter25 +=1
    return TrajSteps


def main(
    FileName,
    NumBasisFunctions,
    rcut_inf,
    rcut_yes,
    r_mod,
    timestep_skip,
    output_file
):   
    
    try:
        # Check if file exists
        datadir = os.path.dirname(os.path.abspath(__file__))
        input_file_path = os.path.join(datadir, FileName)
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
        
        # Load the data and check format
        Cell_df = pd.read_csv(input_file_path)
        
        # Check if required columns exist
        required_columns = ['id', 'timestep', 'x', 'y']
        missing_columns = [col for col in required_columns if col not in Cell_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in input file: {', '.join(missing_columns)}")
        
        # Check for numeric data types
        for col in ['id', 'timestep', 'x', 'y']:
            if not pd.api.types.is_numeric_dtype(Cell_df[col]):
                raise ValueError(f"Column '{col}' must contain numeric values")
                
        # Check parameter constraints
        if NumBasisFunctions <= 0:
            raise ValueError("NumBasisFunctions must be a positive integer")
        if r_mod <= 0:
            raise ValueError("r_mod must be a positive number")
        if timestep_skip <= 0:
            raise ValueError("timestep_skip must be a positive integer")
            
        # Process data as before
        Cell_df = Cell_df.sort_values(by='id').reset_index(drop=True)
        CellInd = Cell_df.index[Cell_df['id'].diff().ne(0)].tolist()
        CellInd.append(len(Cell_df) - 1)
        
        # Continue with the rest of your function
        Cell_df['timestep'] = Cell_df['timestep'].round(2)
        Cell_df = Cell_df.sort_values(by=['id', 'timestep']).reset_index(drop=True)
        
        # Rest of your code...
    
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nExpected input format:")
        print("  - CSV file with columns: id, timestep, x, y")
        print("  - Numeric values for all columns")
        print("  - Multiple timesteps and multiple cell IDs")
        print("\nExample command:")
        print(f"  python {os.path.basename(__file__)} --FileName=data.csv --NumBins=100 --r_range=100")
        return 1  # Return error code

    GetTimesteps = False
    UseDistParams = True
    SaveData = True

    TrajSteps = None
    if GetTimesteps:
        TrajSteps = get_timestepHistogram(Cell_df,CellInd)
        
    resultsFile = []
    if UseDistParams:
        R_NT,R_N1,R_mode,AvNN = Get_Distance_Quantities(Cell_df)
        r_mod = R_N1
        if rcut_yes:
            rcut_inf = R_NT*0.5 #*(3/2)
        else:
            rcut_inf = 1000

    NumBasisFunctions = 10
    OnlyNN = False
    #print("params used: rcut_inf: 1000 r_mod: 22.36508405242504 NumBasisFunctions: 10 timestep_skip: 1
    print(f"Using parameters: rcut_inf={rcut_inf}, r_mod={r_mod}, NumBasisFunctions={NumBasisFunctions}, timestep_skip={timestep_skip}")
    print("r_cut_yes:",rcut_yes)
    #Using parameters: rcut_inf=198.3293332547048, r_mod=22.36508405242504, NumBasisFunctions=10, timestep_skip=1
    FF, Rhist,Bm05 = estmateforcesMultiPairLJ(Cell_df,NumBasisFunctions,OnlyNN,rcut_inf,rcut_yes,r_mod,timestep_skip)

    # Store results in list with all relevant data
    resultsFile.append({
    'Bm05': Bm05.copy(),
    'Force_coeffs': FF.copy(),
    'Rhist': Rhist.copy(),
    'rcut_inf': rcut_inf,
    'rcut_yes': rcut_yes,
    'r_mod': r_mod,
    'file': FileName,
    'R_NT': R_NT,
    'R_N1': R_N1,
    'R_mode': np.mean([R_mode[0],R_mode[1]]),
    'AvNN': AvNN,
    'timestep_skip': timestep_skip,
    'TrajSteps': TrajSteps
    })

    print(f"Completed run")

    # for i, res in enumerate(resultsFile2):
    #     print(f"Run {i}: timestep_skip={skips[i]}, rcut_inf={res['rcut_inf']}, r_mod={res['r_mod']}")


   # In your main function:
    if SaveData:
        # Convert arrays to float type explicitly before saving
        Bm05_float = resultsFile[0]['Bm05'].astype(np.float64)
        Force_coeffs_float = resultsFile[0]['Force_coeffs'].astype(np.float64)
        Rhist_float = np.array(resultsFile[0]['Rhist'], dtype=np.float64)
        
        # Save arrays separately using numpy's save with .dat extension
        np.save(f"{output_file}_Bm05.dat", Bm05_float)
        np.save(f"{output_file}_Force_coeffs.dat", Force_coeffs_float)
        np.save(f"{output_file}_Rhist.dat", Rhist_float)
        
        # Save scalar values to CSV (also with .dat extension)
        scalar_results = {
            'rcut_inf': float(resultsFile[0]['rcut_inf']),
            'rcut_yes': bool(resultsFile[0]['rcut_yes']),
            'r_mod': float(resultsFile[0]['r_mod']),
            'file': str(resultsFile[0]['file']),
            'R_NT': float(resultsFile[0]['R_NT']),
            'R_N1': float(resultsFile[0]['R_N1']),
            'R_mode': float(resultsFile[0]['R_mode']),
            'timestep_skip': int(resultsFile[0]['timestep_skip']),
            'NumBasisFunctions': int(NumBasisFunctions)
        }
        pd.DataFrame([scalar_results]).to_csv(f"{output_file}_params.dat", index=False)
        
        # # Also save a simple pickle file that preserves all data structures
        # with open(f"{output_file}.pkl", 'wb') as f:
        #     pickle.dump(resultsFile, f)
            
        print(f"Saved results to multiple .dat files with base name {output_file}")
        return 0  # Success

if __name__ == '__main__':
    args = parse_args()
    exit_code = main(
        FileName=args.FileName,
        NumBasisFunctions=args.NumBasisFunctions,
        rcut_inf=args.rcut_inf,
        rcut_yes=args.rcut_yes,
        r_mod=args.r_mod,
        timestep_skip=args.timestep_skip,
        output_file=args.output_file
    )
    # Optionally exit with the appropriate code
    import sys
    sys.exit(exit_code)
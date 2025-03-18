import pandas as pd
from bs4 import BeautifulSoup
import glob

def extract_player_data(html_file, output_csv):
    with open(html_file, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
    
    players = []
    
    # Extract player positions and stats
    positions = [pos.get_text(strip=True) for pos in soup.find_all("span", class_="Table_tag__vKZKn generated_utility20sm__ZX2Hf generated_utility19md__XKkU_")]
    stat_elements = soup.find_all("span", class_="Table_statCellValue__zn5Cx")
    stats = [stat.get_text(strip=True)[:2] for stat in stat_elements]
    
    # Assuming each player has 14 stats, split the stats into chunks
    num_stats = 14
    stats_chunks = [stats[i:i+num_stats] for i in range(0, len(stats), num_stats)]
    
    # Ensure data consistency
    for i in range(min(len(positions), len(stats_chunks))):
        players.append([positions[i]] + stats_chunks[i][:7])
    
    columns = ["Position", "Overall", "Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physicality"]
    df = pd.DataFrame(players, columns=columns)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    print(f"Data successfully saved to {output_csv}")

def merge_csv_files(input_folder, output_csv):
    csv_files = glob.glob(f"{input_folder}/*.csv")
    dataframes = [pd.read_csv(file) for file in csv_files]

    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df = merged_df.drop_duplicates()
    merged_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Merged CSV saved as {output_csv}")

def revise_position(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df["Position"] = df["Position"].apply(lambda x: "FW" if x in ["ST", "LW", "RW"] else "MF" if x in ["CM", "CAM", "CDM", "LM", "RM"] else "DF" if x in ["CB", "LB", "RB"] else "GK")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Revised data saved as {output_csv}")

def revise_9position(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df["Position"] = df["Position"].apply(lambda x: "WF" if x in ["LW", "RW"] else "SMF" if x in ["LM", "RM"] else "SB" if x in ["LB", "RB"] else x)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Revised data saved as {output_csv}")

def get_random_players_data(input_csv, output_csv, num_samples):
    df = pd.read_csv(input_csv)
    sampled_df = df.groupby("Position").apply(lambda x: x.sample(n=num_samples, random_state=42, replace=False)).reset_index(drop=True)
    sampled_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Random sampled data saved as {output_csv}")

def get_random_total_players_data(input_csv, output_csv, num_samples):
    df = pd.read_csv(input_csv)
    sampled_df = df.sample(n=num_samples, random_state=42, replace=False)
    sampled_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Random sampled data saved as {output_csv}")

""" extract data """
# for i in range(1, 21):
#     html_file = f"FC25players{i}.html"
#     output_csv = f"players_data{i}.csv"
#     extract_player_data(html_file, output_csv)

""" merge data """
# merge_csv_files("C:\code_sem\senior\AIC\AIC HW1\dataset\data_1-20", "data2000.csv")

""" revise position to 4 categories """
# revise_position("dataset/data2000.csv", "dataset/4categories.csv")

""" revise position to 9 categories """
# revise_9position("dataset/data2000.csv", "dataset/9categories.csv")

""" get 200 players data for each position """
# get_random_players_data("dataset/4categories.csv", "dataset/4cate_200.csv", 200)

""" get 100 players data for each position """
# get_random_players_data("dataset/9categories.csv", "dataset/9cate_100.csv", 100)

""" get random 1000 players data """
get_random_total_players_data("dataset/data2000.csv", "dataset/data1000.csv", 1000)
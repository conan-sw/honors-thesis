import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import re
import hashlib


# Read the individuals CSV files downloaded from Log Explorer, combine them into one DataFrame
sheets = os.listdir("./data/raw_data")
logs = pd.DataFrame()
for sheet in sheets:
  path = "./data/raw_data/" + sheet
  df = pd.read_csv(path)[["textPayload"]]
  logs = pd.concat([logs, df])

logs = logs["textPayload"].reset_index(drop = True)

# Use regex patterns to extract relevant question-answer pairs from form responses
form_text = logs[logs.str.contains(r'sid').fillna(False)].astype(str)
form_df = pd.DataFrame()
form_df["sid"] = form_text.str.extract(r"\'sid\': \'(\d+)\'").fillna(0).astype(int)
form_df["assignments"] = form_text.str.extract(r"\'assignments\': \'([\w\s\(\)/\.,:\-]+)\'")
form_df["timestamp"] = form_text.str.extract(r"\'timestamp\': \'(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{1,2}:\d{1,2})\'")
form_df["timestamp"] = pd.to_datetime(form_df["timestamp"], format = '%m/%d/%Y %H:%M:%S')
form_df["days"] = form_text.str.extract(r"\'days\': \'([\d, ]+)\'")
form_df = form_df.groupby(["sid", "timestamp"]).agg("first").reset_index()
form_df = form_df[(~(form_df["assignments"].isna())) & (form_df["sid"] > 3_000_000_000)]

# Extract SID, timestamp, and class URL to join with logs containing form responses
class_text = logs[logs.str.contains(r'Student ID Number').fillna(False)]
class_df = pd.DataFrame()
class_df["sid"] = class_text.str.extract(r'\"Student ID Number\": \[\"(\d+)\"\]').fillna(False).astype(int)
class_df["timestamp"] = class_text.str.extract(r'\"Timestamp\": \[\"(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{1,2}:\d{1,2})\"\]')
class_df["timestamp"] = pd.to_datetime(class_df["timestamp"], format = '%m/%d/%Y %H:%M:%S')
class_df["url"] = class_text.str.extract(r'\"spreadsheet_url\": \"(.*)\", \"form_data\"')

# Remove duplicates under the assumption that each (SID, timestamp) pair points to a unique response
class_df = class_df.groupby(["sid", "timestamp"]).agg("first").reset_index()

# Map form responses to corresponding class URLs 
df = pd.merge(class_df, form_df, on = ["sid", "timestamp"], how = "inner")

# Read in spreadsheet mapping class URLs to class names
urls = pd.read_csv("./data/classes.csv")

# Read in spreadsheet mapping assignments to deadlines
deadlines = pd.read_csv("./data/deadlines.csv")

# Separate responses containing multiple assignments into individual rows
df["assignments"] = df["assignments"].str.split(",")
df = df.explode("assignments")

# Map class URLs to class names, map assignments to deadlines
df_urls = df.merge(urls, left_on = "url", right_on = "Spreadsheet link")
master = df_urls.merge(deadlines, left_on = "Course Number", right_on = "Class Name")

# Filter down to matching assignment types
bools = ((master["assignments"].str.split().str[0] == master["Assignment Category"]) | (master["Assignment Number within Category"] == master["assignments"].str.split().str[0]))
master = master[bools]

# Hash student ID, ensure the process retains uniqueness and one-to-one mapping
master["hashed_sid"] = master["sid"].map(lambda s: hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:6])
assert len(master["hashed_sid"].unique()) == len(master["sid"].unique())
master.drop(axis=1, columns=["sid"], inplace=True)

# Identify number of students enrolled in more than one class
master.groupby("sid")["url"].agg(lambda x: len(x.unique())).value_counts()

# Create list of booleans to match form responses to correct assignment
more_bools = []
for i in np.arange(master.shape[0]):
  row = master.iloc[i, :]
  if re.search("Homework", row["Assignment Category"]):
    if not pd.isnull(row["Subassignment number"]):
      if (bool(re.search(row["Subassignment number"], row["assignments"])) & bool(re.search(row["Assignment Number within Category"], row["assignments"]))):
        more_bools.append(True)
      else:
        more_bools.append(False)
    else:
      if re.search(row["Assignment Number within Category"], row["assignments"]):
        more_bools.append(True)
      else:
        more_bools.append(False)
  elif re.search("Lab", row["Assignment Category"]):
    if re.search(r'\d\.\d', row["assignments"]):
      num = row["assignments"].split()[1]
      if ((row["Subassignment number"] == row["assignments"].split(".")[1]) & (row["Assignment Number within Category"] == row["assignments"].split(".")[0])):
        more_bools.append(True)
      else:
        more_bools.append(False)
    else:
      if re.search(row["Assignment Number within Category"], row["assignments"]):
        more_bools.append(True)
      else:
        more_bools.append(False)
  elif re.search("Project", row["Assignment Category"]):
    if re.search("Project", row["assignments"]):
      if not pd.isnull(row["Subassignment number"]):
        if (bool(re.search(row["Subassignment number"], row["assignments"])) & bool(re.search(row["Assignment Number within Category"], row["assignments"]))):
          more_bools.append(True)
        else:
          more_bools.append(False)
      else:
        if re.search(row["Assignment Number within Category"], row["assignments"]):
          more_bools.append(True)
        else:
          more_bools.append(False)
    elif re.search(str(row["Subassignment number"]), row["assignments"]):
      more_bools.append(True)
    elif (bool(re.search('Final Deadline', row["assignments"])) & pd.isnull(row["Subassignment number"])):
      more_bools.append(True)
    else:
      more_bools.append(False)
  elif re.search("WSP", row["Assignment Category"]):
    if re.search("WSP", row["assignments"]):
      if re.search(row["Assignment Number within Category"], row["assignments"]):
        more_bools.append(True)
      else:
        more_bools.append(False)
  else:
    more_bools.append(False)
final_master = master[more_bools]

# Keep relevant columns, convert to appropriate data types in preparation for analysis
df = final_master[["hashed_sid", "timestamp", "assignments", "days", "Class Name", "Class Type", "Assignment Category", "Assignment Number within Category", "Subassignment number", "Regular Deadline"]]
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["Regular Deadline"] = pd.to_datetime(df["Regular Deadline"])
df["Assignment Category"] = df["Assignment Category"].replace({"WSP": "Homework"})
df["delta"] = df["timestamp"] - df["Regular Deadline"]
df["delta (min)"] = df["delta"].astype(int) / 10**9/60

# Export resulting DataFrame as "cleaned_data.csv". This file should be placed under "./data" for analysis
df.to_csv("cleaned_data.csv")

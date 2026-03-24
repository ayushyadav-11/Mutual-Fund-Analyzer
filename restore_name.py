import json

with open("d:/Mutual Fund Analyzer/session_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

if "investor_info" in data:
    data["investor_info"]["name"] = "Anita Yadav" # Assuming from the previous snapshot

with open("d:/Mutual Fund Analyzer/session_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Session data restored name.")

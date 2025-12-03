import json


def write_jsonl_file(json_msg, output_filename: str):
    with open(output_filename, "w", encoding="utf-8") as jsonl_file:
        for item in json_msg:
            line = json.dumps(item, ensure_ascii=False)
            jsonl_file.write(line + "\n")


def write_json_file(json_msg, output_filename: str):
    with open(output_filename, "w", encoding="utf-8") as json_file:
        json.dump(json_msg, json_file, ensure_ascii=False, indent=2)


def load_json_file(file_path) -> json:
    with open(file_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def load_jsonl_file(file_path) -> list:
    jsons = []

    with open(file_path, "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            line = line.strip()

            if not line:
                continue

            jsons.append(json.loads(line))

    return jsons

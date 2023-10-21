import ast
import json
import argparse

def read_source_code(file_path):
    with open(file_path, "r") as f:
        return f.read()

def write_json_to_file(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def extract_relations(source_code):
    tree = ast.parse(source_code)
    relations = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            relations[class_name] = {}
            local_methods = set()  # Keep track of local methods
            
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.FunctionDef):
                    method_name = sub_node.name
                    local_methods.add(method_name)  # Add to local methods set
                    relations[class_name][method_name] = {"calls": [], "length": len(sub_node.body)}

            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.FunctionDef):
                    method_name = sub_node.name
                    for method_node in ast.walk(sub_node):
                        if isinstance(method_node, ast.Call):
                            called_method = None
                            if isinstance(method_node.func, ast.Name):
                                called_method = method_node.func.id
                            elif isinstance(method_node.func, ast.Attribute):
                                called_method = method_node.func.attr

                            if called_method in local_methods and called_method not in relations[class_name][method_name]["calls"]:
                                relations[class_name][method_name]["calls"].append(called_method)
                        
                        # if isinstance(method_node, ast.Name):
                        #     relations["classes"][class_name]["methods"][method_name]["variables"].append(method_node.id)
                            
    return relations

def generate_mermaid_diagram(data):
    mermaid_lines = []
    mermaid_lines.append("graph TD")

    for key, values in data.items():
        calls = values["calls"]
        for value in calls:
            mermaid_lines.append(f"    {key} --> {value}")

    return "\n".join(mermaid_lines)

def main():
    parser = argparse.ArgumentParser(description="Extract relations from Python code.")
    parser.add_argument("input", nargs="?", help="Path to the input Python file.")
    parser.add_argument("output", nargs="?", help="Path to the output JSON file.")
    args = parser.parse_args()

    if not args.input or not args.output:
        print("Both input and output file paths are required.")
        exit(1)

    source_code = read_source_code(args.input)
    relations = extract_relations(source_code)
    write_json_to_file(relations, args.output)
    
    first_class = list(relations.keys())[0]
    functions = relations[first_class]
    mermaid_diagram = generate_mermaid_diagram(functions)
    mermaid_filename = args.output.replace(".json", ".txt")
    with open(mermaid_filename, "w") as f:
        f.write(mermaid_diagram)


if __name__ == "__main__":
    main()

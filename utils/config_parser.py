import configparser


def parse_config(file_name: str):
    # append path in docker to file_name if not provided
    if "/" not in file_name:
        file_name = f"/workspace/configurations/{file_name}"
    config = configparser.ConfigParser()
    config.read(file_name)
    return config
    


if __name__ == "__main__":
    # try parsing file
    file_name = "/workspace/configurations/bigram.ini"
    config = parse_config(file_name=file_name)
    print(config)
    print(config.sections())
    print(config["Model"])
    print(config["Dataset"])
    print(config["Tokenizer"])
    print(config["Model"]["n_embed"])
    
        
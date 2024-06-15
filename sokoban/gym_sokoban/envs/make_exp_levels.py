import os

def make_rotations(level_lines):
    rotate_lines_1 = []
    rotate_lines_2 = []
    rotate_lines_3 = []
    for y in range(10):
        new_line_1 = []
        new_line_2 = []
        new_line_3 = []
        for x in range(10):
            new_line_1.append(level_lines[-(x+1)][y])
            new_line_2.append(level_lines[-(y+1)][-(x+1)])
            new_line_3.append(level_lines[x][-(y+1)])
        rotate_lines_1.append(new_line_1)
        rotate_lines_2.append(new_line_2)
        rotate_lines_3.append(new_line_3)
    return (level_lines, rotate_lines_1, rotate_lines_2, rotate_lines_3)

def process_levels(levels):
    all_levels = []
    for j in range(len(levels.split(";")[1:])):
        level_strings = levels.split(";")[1:][j].split("\n")[1:]
        if level_strings[-1] == "":
            level_strings = level_strings[:-1]
        raw_lines, mirror_lines = [], []
        for line in level_strings:
            line_list = list(line)
            mirror_list = []
            for i in range(len(line)):
                mirror_list.append(line_list[-(i+1)])
            raw_lines.append(line_list)
            mirror_lines.append(mirror_list)
        all_levels += make_rotations(raw_lines)
        all_levels += make_rotations(mirror_lines)
    return all_levels

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description="create clean+corrupted experiment Sokoban levels")
    parser.add_argument("--expname", type=str, required=True)
    args = parser.parse_args()

    expname = args.expname
    print(f"Creating clean+corrupted Sokoban levels for {expname}")

    with open(f"./exp-levels-txt/{expname}/clean.txt") as f:
        clean_levels = f.read()
    with open(f"./exp-levels-txt/{expname}/corrupt.txt") as f:
        corrupt_levels = f.read()

    all_clean_levels = process_levels(clean_levels)
    all_corrupt_levels = process_levels(corrupt_levels)

    level_id = 0
    exp_dir = f"./boxoban-levels/experiments/{expname}"
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    for clean_level, corrupt_level in zip(all_clean_levels, all_corrupt_levels):
        level_dir = exp_dir + f"/{level_id:04}"
        if not os.path.exists(level_dir):
            os.mkdir(level_dir)
            os.mkdir(level_dir+"/clean")
            os.mkdir(level_dir+"/corrupt")
        clean_level = ["".join(line) for line in clean_level]
        clean_level = [f"; {level_id}"] + clean_level
        clean_level = "\n".join(clean_level)
        with open(level_dir+"/clean/000.txt", "w") as f:
            f.write(clean_level)
        corrupt_level = ["".join(line) for line in corrupt_level]
        corrupt_level = [f"; {level_id}"] + corrupt_level
        corrupt_level = "\n".join(corrupt_level)
        with open(level_dir+"/corrupt/000.txt", "w") as f:
            f.write(corrupt_level)
        level_id += 1